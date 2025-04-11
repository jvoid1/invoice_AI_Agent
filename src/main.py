import os
import re
import asyncio
import platform
import base64
from getpass import getpass
from typing_extensions import TypedDict
from typing import List, Optional
from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page
from playwright.async_api import async_playwright
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from datetime import datetime
from pathlib import Path
import json
import logging
from PIL import Image
import io

# Function to get OpenAI API key
def _getpass(env_var: str):
    if not os.environ.get(env_var):
        os.environ[env_var] = getpass(f"{env_var}=")

_getpass("OPENAI_API_KEY")

# Define bounding box and prediction types
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    context: List[BaseMessage]
    observation: str
    initial_url: str

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup file handler
    log_file = log_dir / f"invoice_AI_Agent_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('invoice_AI_Agent')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Define tools
async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    logger = logging.getLogger('invoice_AI_Agent')
    
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    
    bbox_id = int(click_args[0])
    try:
        # Initial number of pages
        context = page.context
        initial_pages = len(context.pages)
        
        bbox = state["bboxes"][bbox_id]
        target_text = bbox.get("text") or bbox.get("ariaLabel") or ""
        
        # Setup event listener for new page
        async def handle_new_page(page):
            logger.info(f"Handler intialized")
            
        context.on("page", handle_new_page)
               
        # Click in main page
        x, y = bbox["x"], bbox["y"]
        await page.mouse.click(x, y)
        await asyncio.sleep(3)
        
        # Check if new tab was opened
        current_pages = len(context.pages)
        if current_pages > initial_pages:
            new_page = context.pages[-1]
            logger.info(f"New tab detected after clicking {bbox_id}. URL: {new_page.url}")
        
        await sleep(10) 
    except Exception as e:
        return f"Error clicking bbox {bbox_id}: {str(e)}"
        
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    logger = logging.getLogger('invoice_AI_Agent')
    
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    
    try:
        logger.info("Starting keyboard navigation sequence")
        typing_completed = False
        
        # 1. Initial tab sequence with verification
        logger.info("Initializing form navigation")
        for i in range(4):
            await page.keyboard.press("Tab")
            await page.wait_for_timeout(500)
            try:
                # Verify if we can interact with the current element
                focused = await page.evaluate('document.activeElement !== document.body')
                if not focused:
                    logger.warning(f"Tab {i+1} did not focus an element, retrying...")
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error verifying focus: {str(e)}")
        
        # Field sequence with verification
        form_sequence = [
            ("123456", 1),      # Order number
            ("789.01", 1),      # Amount 1
            ("2345.67", 1),     # Amount 2
            ("test010101abc", 1),  # RFC
            ("test user", 1),    # Name
            ("test@example.com", 1),  # Email
            ("89012", 1),       # CP
            (None, 2),          # Final tabs
        ]
        
        field_completion = []
        
        for idx, (content, tab_count) in enumerate(form_sequence):
            if content:
                field_typed = False
                retry_count = 0
                max_retries = 3
                
                while not field_typed and retry_count < max_retries:
                    try:
                        logger.info(f"Typing field {idx + 1}: {content}")
                        
                        await page.keyboard.press("Control+a")
                        await page.wait_for_timeout(300)
                        await page.keyboard.press("Backspace")
                        await page.wait_for_timeout(300)
                        
                        # Type content with verification
                        for char in content:
                            await page.keyboard.type(char)
                            await page.wait_for_timeout(100)  # Increased delay between chars
                        
                        focused_value = await page.evaluate('document.activeElement.value')
                        if focused_value == content:
                            field_typed = True
                            field_completion.append(True)
                            logger.info(f"Successfully typed and verified: {content}")
                        else:
                            logger.warning(f"Content verification failed. Expected: {content}, Got: {focused_value}")
                            retry_count += 1
                    except Exception as e:
                        logger.error(f"Error typing field {idx + 1}: {str(e)}")
                        retry_count += 1
                        await asyncio.sleep(1)
                
                if not field_typed:
                    logger.error(f"Failed to type field {idx + 1} after {max_retries} attempts")
                    return f"Failed to type field {content} after multiple attempts"
            
            # Tab to next field with verification
            for _ in range(tab_count):
                await page.keyboard.press("Tab")
                await page.wait_for_timeout(500)
                
        # Press Enter after final tab sequence
        logger.info("Pressing Enter to submit form")
        await page.keyboard.press("Enter")
        await asyncio.sleep(1)
        
        # Verify all fields were completed
        typing_completed = len(field_completion) == len([x for x, _ in form_sequence if x])
        
        if typing_completed:
            logger.info("Form filling sequence fully completed and verified")
            await asyncio.sleep(2)  
            return "Form filling sequence completed successfully"
        else:
            logger.error("Form filling sequence incomplete")
            return "Form filling sequence did not complete"
        
    except Exception as e:
        error_msg = f"Error during keyboard navigation: {str(e)}"
        logger.error(error_msg)
        return error_msg

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."
    target, direction = scroll_args
    scroll_amount = 500 if target.upper() == "WINDOW" else 200
    scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
    if target.upper() == "WINDOW":
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)
    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_home(state: AgentState):
    page = state["page"]
    initial_url = state.get("initial_url", "https://www.google.com")
    await page.goto(initial_url)
    return f"Navigated back to {initial_url}"

async def select_option(state: AgentState):
    page = state["page"]
    select_args = state["prediction"]["args"]
    
    if select_args is None or len(select_args) != 2:
        return "Failed to select option due to incorrect arguments"
        
    bbox_id = int(select_args[0])
    option_text = select_args[1]
    
    try:
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        
        await page.wait_for_timeout(1000)
        await page.mouse.click(x, y)
        await page.wait_for_timeout(1000)
        
        success = False
        
        # Strategy 1: Try native select element
        try:
            select_selector = f'select, [role="combobox"], [role="listbox"]'
            select_element = await page.query_selector(f'{select_selector}:near({x}, {y}, 50)')
            if select_element:
                await select_element.select_option(label=option_text)
                await page.wait_for_timeout(1000)
                success = True
        except Exception:
            pass

        # Strategy 2: Try clicking visible option
        if not success:
            try:
                await page.wait_for_selector('[role="option"], [role="listitem"], .select-option', timeout=2000)
                option_selectors = [
                    f'text="{option_text}"',
                    f'[role="option"]:has-text("{option_text}")',
                    f'[role="listitem"]:has-text("{option_text}")',
                    f'.select-option:has-text("{option_text}")'
                ]
                for selector in option_selectors:
                    try:
                        await page.click(selector, timeout=1000)
                        success = True
                        break
                    except:
                        continue
            except Exception:
                pass

        # Strategy 3: JavaScript injection approach
        if not success:
            try:
                js_script = f"""
                (function() {{
                    const target = document.elementFromPoint({x}, {y});
                    if (target) {{
                        // Handle select element
                        if (target.tagName === 'SELECT') {{
                            for (let option of target.options) {{
                                if (option.text.includes('{option_text}')) {{
                                    target.value = option.value;
                                    target.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                    return true;
                                }}
                            }}
                        }}
                        // Handle custom select/dropdown
                        target.value = '{option_text}';
                        target.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        return true;
                    }}
                    return false;
                }})()
                """
                success = await page.evaluate(js_script)
            except Exception:
                pass

        await page.wait_for_timeout(2000)
        
        if success:
            return f"Selected option '{option_text}' from dropdown {bbox_id}"
        else:
            # Try one last time with a simple click
            await page.mouse.click(x, y)
            await page.keyboard.type(option_text)
            await page.keyboard.press("Enter")
            return f"Attempted to select '{option_text}' using fallback method"
            
    except Exception as e:
        return f"Error selecting option from bbox {bbox_id}: {str(e)}"

async def optimize_screenshot(screenshot_bytes, max_size=(800, 800)):
    """Optimize screenshot size and quality for GPT-4V"""
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(screenshot_bytes))
    
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    
    # Determine new dimensions while maintaining aspect ratio
    if image.width > max_size[0] or image.height > max_size[1]:
        if aspect_ratio > 1:
            new_width = min(image.width, max_size[0])
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(image.height, max_size[1])
            new_width = int(new_height * aspect_ratio)
        
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to bytes with reduced quality
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=80, optimize=True)
    return output.getvalue()

# Define the mark_page function
@chain
async def mark_page(page):
    with open("mark_page.js") as f:
        mark_page_script = f.read()
    
    bboxes = []
    try:
        # Check iframes
        iframes = await page.query_selector_all('iframe')
        current_index = 0  
        
        for iframe in iframes:
            frame = await iframe.content_frame()
            if frame:
                await frame.evaluate(mark_page_script)
                iframe_bboxes = await frame.evaluate("markPage()")
                if iframe_bboxes and len(iframe_bboxes) > 0:
                    # Modify bbox indices to continue from current_index
                    for bbox in iframe_bboxes:
                        bbox['index'] = current_index  # Add index to track original position
                        current_index += 1
                    bboxes.extend(iframe_bboxes)
                    screenshot = await page.screenshot()
                    break
        
        # If no elements found in iframes, or if we want to include main page elements
        if len(bboxes) == 0:
            await page.evaluate(mark_page_script)
            main_bboxes = await page.evaluate("markPage()")
            # Modify main page bbox indices
            for bbox in main_bboxes:
                bbox['index'] = current_index
                current_index += 1
            bboxes.extend(main_bboxes)
            screenshot = await page.screenshot()
    
    except Exception as e:
        print(f"Error marking page: {str(e)}")
        await asyncio.sleep(8)
        return await mark_page(page)  # Retry
            
    optimized_screenshot = await optimize_screenshot(screenshot)
    
    # Save optimized screenshot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    screenshots_dir = Path("recordings") / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshots_dir / f"optimized_{timestamp}.jpg"
    Image.open(io.BytesIO(optimized_screenshot)).save(screenshot_path)
    
    # Cleanup
    await page.evaluate("unmarkPage()")
    for iframe in await page.query_selector_all('iframe'):
        frame = await iframe.content_frame()
        if frame:
            await frame.evaluate("unmarkPage()")
            
    return {
        "img": base64.b64encode(optimized_screenshot).decode(),
        "bboxes": bboxes,
    }

# Define agent functions
async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state):
    labels = []
    for bbox in state["bboxes"]:
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        index = bbox.get("index", 0)
        labels.append(f'{index} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]
    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}

async def switch_to_iframe_if_needed(page, target_text):
    """Switch to iframe if it contains the target element"""
    # Get all iframes
    iframes = await page.query_selector_all('iframe')
    
    # Store original context
    main_frame = page.main_frame
    
    for iframe in iframes:
        frame = await iframe.content_frame()
        if frame:
            # Switch to iframe context
            await page.wait_for_selector('iframe')
            frame = await page.frame_locator('iframe').first
            
            # Find element in iframe
            element = await frame.get_by_text(target_text, exact=False).count()
            if element > 0:
                return frame
            
    return main_frame

SYSTEM_PROMPT = """Imagine you are a robot browsing the web, just like humans. You are now assigned a specific task: generate an electronic invoice (factura) on a Mexican webpage. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot features Numerical Labels in the TOP LEFT corner of each Web Element, used for identifying interaction targets.
Your goal is to complete the invoicing (facturación) process. Carefully analyze the page to:

1. Locate and click "Facturación", "Factura", or select a store or location from a menu.
2. Navigate to the invoice form and fill all required fields with provided information.
3. Look for and click a button to submit the form, typically labeled "Emitir Factura", "Enviar", or "Facturar".
4. If any lightboxes, pop-ups, or cookie notices appear that block interaction, close or skip them.
5. Avoid clicking login, register, donation, or unrelated ads.

Action should STRICTLY follow the format:

- Click [Numerical_Label]
- Type [Numerical_Label]; [content]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Select [Numerical_Label]; [Option_Text]
- Wait
- GoBack
- Home
- ANSWER; [content]

Key Guidelines:
- One action per iteration only.
- Always select the correct bounding box based on the label in the top-left corner.
- Avoid wasting time on unrelated elements (login, ads, etc).
- Close lightboxes or pop-ups that block the screen and progress.

Task-Specific Heuristics:
- Look for keywords like: Factura, Facturación, Tienda, Emitir, RFC, Ticket, Enviar, Emitir Factura.
- In dropdown or list views, choose the correct store or purchase source if required.
- In form fields, match placeholder or label text like RFC, Uso de CFDI, Número de ticket, Fecha, etc.
- Look for the final action button with texts such as Facturar, Emitir, Generar factura, or Enviar.
- If a popup appears (cookie, newsletter, ad), find and click Cerrar, X, or Aceptar to close it.

Your reply must strictly follow this format:
Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}
Action: {{One Action format you choose}}"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="context"),
    ("human", "![image](data:image/png;base64,{img})\n{bbox_descriptions}\n{input}")
])

# Initialize the LLM and agent
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

def update_context(state: AgentState):
    """After a tool is invoked, we want to update
    the context so the agent is aware of its previous steps"""
    old = state.get("context")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "context": [SystemMessage(content=txt)]}

# Compile the graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("update_context", update_context)
graph_builder.add_edge("update_context", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Home": to_home,
    "Select": select_option,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_context")

def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

# Use the graph
async def call_agent(question: str, page, max_steps: int = 20, is_new_tab: bool = False, previous_page: Page = None):
    logger = logging.getLogger('invoice_AI_Agent')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_new_tab:
        logger.info("Closing other tabs before starting new tab session")
        context = page.context
        current_pages = context.pages
        
        for other_page in current_pages:
            if other_page != page:
                try:
                    logger.info(f"Closing tab with URL: {other_page.url}")
                    await other_page.close()
                except Exception as e:
                    logger.error(f"Error closing tab: {str(e)}")
    
    # Log session start with tab context
    if is_new_tab:
        logger.info(f"Starting new tab session at URL: {page.url}")
        logger.info("Using invoice form specific prompt")
    else:
        logger.info(f"Starting initial session at URL: {page.url}")
    
    # Create directory for recordings
    recordings_dir = Path("recordings") / timestamp
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    # Select appropriate system prompt based on context
    if is_new_tab:
        system_prompt = """You are now in the invoice generation page. Your task is to fill out the form with the following test data:
        
        Fill in these required fields:
        - RFC: TEST010101ABC
        - Email: test@example.com
        - Order Number: 123456 
        - Total Amount: 1234.56
        - CFDI Usage: G03 - Gastos en general
        
        Look for:
        - Input fields for RFC, email, order details
        - Dropdown menus for CFDI usage
        - Submit/Generate invoice button
        
        Action should STRICTLY follow the format:
        - Type [Numerical_Label]; [content]
        
        Your reply must follow this format:
        Thought: I see [describe what field you're targeting]
        Action: [SINGLE action in correct format]"""
    else:
        system_prompt = SYSTEM_PROMPT
    
    # Create prompt template with selected system prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="context"),
        ("human", "![image](data:image/png;base64,{img})\n{bbox_descriptions}\n{input}")
    ])
    
    # Initialize agent with new prompt
    agent = annotate | RunnablePassthrough.assign(
        prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
    )
    
    # Recompile graph with new agent
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_node("update_context", update_context)
    graph_builder.add_edge("update_context", "agent")
    
    for node_name, tool in tools.items():
        graph_builder.add_node(
            node_name,
            RunnableLambda(tool) | (lambda observation: {"observation": observation}),
        )
        graph_builder.add_edge(node_name, "update_context")

    graph_builder.add_conditional_edges("agent", select_tool)
    graph = graph_builder.compile()
    
    # Record initial state
    session_record = {
        "timestamp": timestamp,
        "question": question,
        "steps": [],
        "screenshots": [],
        "is_new_tab": is_new_tab,
        "url": page.url
    }
    
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "context": [],
            "initial_url": page.url,
        },
        {
            "recursion_limit": max_steps,
        },
    )
    
    final_answer = None
    steps = []
    new_tab_url = None
    screenshot_path = None 
    
    async for event in event_stream:
        if "agent" not in event:
            continue
            
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        
        # Log each action with tab context
        logger.info(f"{'[NEW TAB] ' if is_new_tab else ''}Step {len(steps) + 1}: {action} - {action_input}")
        print(f"\n{'[NEW TAB] ' if is_new_tab else ''}Step {len(steps) + 1}: {action} - {action_input}")
        
        # For Type actions, ensure completion before continuing
        if action == "Type":
            logger.info("Type action detected - waiting for completion")
            result = await type_text({"page": page, "prediction": pred})
            
            if "failed" in result.lower() or "error" in result.lower():
                logger.error("Type action failed - stopping sequence")
                break
                
            await asyncio.sleep(2)
            logger.info("Type action completed successfully - continuing sequence")
            final_answer = "Form filling completed successfully"
            break
            
        screenshot_path = recordings_dir / f"step_{len(steps):03d}.png"
        await page.screenshot(path=str(screenshot_path))
        
        # Check for new tab after each action
        current_pages = len(page.context.pages)
        if current_pages > 1:
            new_page = page.context.pages[-1]
            new_tab_url = new_page.url
            logger.info(f"New tab detected, will continue processing at: {new_tab_url}")
            
            # Record final step with screenshot path already set
            step_info = {
                "step": len(steps) + 1,
                "action": action,
                "input": action_input,
                "url": page.url,
                "screenshot": str(screenshot_path),
                "new_tab_detected": True,
                "new_tab_url": new_tab_url
            }
            session_record["steps"].append(step_info)
            
            record_file = recordings_dir / "session.json"
            with open(record_file, "w") as f:
                json.dump(session_record, f, indent=2)
            
            logger.info(f"Session recorded to {recordings_dir} before switching to new tab")
            break
        
        # Handle action waits
        if action == "Select":
            await page.wait_for_load_state("networkidle")
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(8)
            await page.wait_for_timeout(3000)
        else:
            await page.wait_for_load_state("networkidle")
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(5)
        
        # Record step information
        step_info = {
            "step": len(steps) + 1,
            "action": action,
            "input": action_input,
            "url": page.url,
            "screenshot": str(screenshot_path),
            "is_new_tab": is_new_tab
        }
        
        session_record["steps"].append(step_info)
        logger.info(f"Step {len(steps) + 1}: {action} - {action_input}")
        
        print(f"\n{len(steps) + 1}. {action}: {action_input}")
        steps.append(step_info)
        if "ANSWER" in action:
            final_answer = action_input[0]
            break
    
    # If new tab was detected, start new agent process with new prompt
    if new_tab_url:
        logger.info(f"Starting new agent process for tab: {new_tab_url}")
        new_page = page.context.pages[-1]
        await new_page.wait_for_load_state("networkidle")
        
        # Close previous tab if it exists
        if previous_page:
            try:
                logger.info("Closing previous tab")
                await previous_page.close()
            except Exception as e:
                logger.error(f"Error closing previous tab: {str(e)}")
        
        return await call_agent(
            """Look at the form fields. Find the first input field in blank
            to start typing the order number by giving the action Type in the
            format: Type [Numerical_Label]""",
            new_page,
            max_steps,
            is_new_tab=True  
        )
    
    # Save final session record
    record_file = recordings_dir / "session.json"
    with open(record_file, "w") as f:
        json.dump(session_record, f, indent=2)
    
    logger.info(f"Session recorded to {recordings_dir}")
    return final_answer

# Main execution
async def main():
    logger = setup_logging()
    logger.info("Starting invoice AI Agent")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False,
                args=['--disable-auto-maximize-for-tests', '--start-fullscreen']
                )
            context = await browser.new_context(
                record_video_dir="recordings/videos",
                viewport={"width": 1920, "height": 1080}
            )
            # Start tracing
            await context.tracing.start(
                screenshots=True,
                snapshots=True
            )
            
            page = await context.new_page()
            await page.goto("https://www.monparis.mx/")
            
            logger.info("Starting agent with question")
            res = await call_agent(
                """Look at the screenshot and find a element with the text 
                'CUMBRES' and click it""", 
                page,
                previous_page=None,
                max_steps=10
            )
            logger.info(f"Final response: {res}")
            
            # Stop tracing
            await context.tracing.stop(path="recordings/trace.zip")
            await context.close()
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("All sessions ended")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
    