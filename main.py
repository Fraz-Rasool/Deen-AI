import os
import asyncio
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple
from nicegui import ui, app
import time

# Global state for dark mode and chat history
dark_mode = False
chat_history: list[Tuple[str, str]] = []

# Import your existing helper functions
try:
    from quran_helper import user_query
    from sahih_bhukari_helper import user_query_sahi_bukhari
    from merger_helper import unified_query
    from sahih_muslim_helper import user_query_sahi_muslim
except ImportError:
    def user_query(question): 
        time.sleep(2)
        return f"Quran guidance regarding '{question}': This is where the authentic Quranic verses and their interpretations would appear based on your question."
    def user_query_sahi_bukhari(question): 
        time.sleep(2)
        return f"Sahih Bukhari hadith for '{question}': Here you would find relevant authentic hadith from Bukhari collection with references."
    def user_query_sahi_muslim(question): 
        time.sleep(2)
        return f"Sahih Muslim hadith for '{question}': Here you would find relevant authentic hadith from Muslim collection with references."
    def unified_query(question): 
        time.sleep(1)
        return f"*Summary for '{question}':*\n\nBased on Islamic sources, here is a comprehensive answer that combines insights from the Quran, Sahih Bukhari, and Sahih Muslim to provide you with authentic Islamic guidance on this topic."

# Load environment variables
load_dotenv()
gem_api = os.getenv("GEMINI_API_KEY")
pinecone_api = os.getenv("PINECONE_API_KEY")

# Add global CSS to remove default margins and padding
ui.add_head_html("""
    <style>
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            overflow-x: hidden;
        }
    </style>
""")

# Backend Functions
def query_all_sources(question: str) -> Dict[str, str]:
    """Query all Islamic sources and return results"""
    return {
        "quran": user_query(question),
        "sahih_bukhari": user_query_sahi_bukhari(question),
        "sahih_muslim": user_query_sahi_muslim(question)
    }

async def process_islamic_query(question: str) -> Dict[str, any]:
    """Process Islamic query asynchronously"""
    try:
        if not question.strip():
            return {
                "success": False,
                "error": "Question cannot be empty"
            }
        
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, unified_query, question)
        sources = await loop.run_in_executor(None, query_all_sources, question)
        
        return {
            "success": True,
            "summary": summary,
            "quran": sources["quran"],
            "sahih_bukhari": sources["sahih_bukhari"],
            "sahih_muslim": sources["sahih_muslim"]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# NiceGUI Frontend Pages
@ui.page('/')
def landing():
    """Landing page with video background"""
    with ui.element('div').classes('fixed inset-0 -z-10 overflow-hidden'):
        ui.video('kabba_2.mp4', autoplay=True, muted=True, loop=True).classes(
            'w-full h-full object-cover'
        ).style('filter: blur(3px); transform: scale(1.05);')
        
    with ui.column().classes('absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center text-center p-4'):
        ui.label('Welcome to DeenAI').classes('text-4xl md:text-6xl font-bold text-[#f5d596] mb-4').style('text-shadow: 1px 1px 5px rgba(0,0,0,0.7);')

        ui.label('Ask anything about Islam, powered by authentic sources from the Quran, Sahih Bukhari, and Sahih Muslim.').classes(
            'text-base md:text-xl mb-8 text-[#f8f8f8] leading-relaxed'
        ).style('text-shadow: 1px 1px 4px rgba(0,0,0,0.6);')

        with ui.row().classes('gap-4 flex-wrap justify-center'):
            ui.link('Enter Chat', '/chat').classes(
                'bg-[#f5d596] bg-opacity-80 hover:bg-opacity-100 font-bold text-[#020a37] px-8 py-4 rounded-full shadow-lg text-lg transition-all duration-300 no-underline'
            ).style('box-shadow: 0 8px 24px rgba(0,0,0,0.3);')
            
            ui.link('About', '/about').classes(
                'bg-transparent border-2 border-[#f5d596] hover:bg-[#f5d596] font-bold text-[#f5d596] hover:text-[#020a37] px-8 py-4 rounded-full shadow-lg text-lg transition-all duration-300 no-underline'
            )

@ui.page('/about')
def about():
    """About page explaining the application"""
    with ui.column().classes(f'min-h-screen w-full p-4 {'bg-[#1a2a2a] text-white' if dark_mode else 'bg-gradient-to-br from-[#f8f8f8] to-[#e6d8c4]'}'):
        with ui.row().classes('fixed top-0 left-0 w-full justify-between items-center shadow-md z-10 h-16 px-4').style('background-color: #0e5449;'):
            with ui.row().classes('items-center gap-3'):
                ui.image('Logo1.png').classes('w-12 h-12')
                ui.label('DeenAI').classes('text-2xl font-bold text-[#f5d596]')
            
            with ui.row().classes('gap-2 items-center'):
                ui.link('🏠 Home', '/').classes('text-base text-[#f5d596] hover:text-white no-underline px-1 py-1 rounded transition-colors')
                ui.link('ℹ About', '/about').classes('text-base text-[#f5d596] hover:text-white no-underline px-3 py-1 rounded transition-colors')
                toggle_label = ui.label('🌙' if dark_mode else '☀').classes('text-2xl cursor-pointer ml-2')
                toggle_label.on('click', lambda: toggle_dark_mode())

        with ui.column().classes('w-full max-w-screen-lg mx-auto mt-20 p-8'):
            ui.label('About DeenAI').classes(f'text-3xl font-bold {'text-[#f8f8f8]' if dark_mode else 'text-[#020a37]'} mb-6')
            
            ui.markdown("""
            *DeenAI* is an Islamic knowledge assistant that provides authentic references from three primary sources:
            
            ### Sources:
            - *The Holy Quran* - The final revelation from Allah (SWT)
            - *Sahih Bukhari* - One of the most authentic hadith collections
            - *Sahih Muslim* - Another highly authentic hadith collection
            
            ### Features:
            - Unified summary combining insights from all sources
            - Individual detailed references from each source
            - Real-time responses powered by AI
            - User-friendly chat interface
            
            ### How to Use:
            1. Click "Enter Chat" to start asking questions
            2. Type your Islamic question in the input box
            3. Receive comprehensive answers with authentic references
            4. Continue the conversation for follow-up questions
            
            May Allah guide us all to the right path. Barakallahu feek!
            """).classes(f'text-{'#d1d5db' if dark_mode else '#1a3a5f'} text-base leading-relaxed')
            
            ui.link('Start Chatting', '/chat').classes(
                'bg-[#0e5449] hover:bg-[#0c4a40] text-white px-6 py-3 rounded-lg font-semibold no-underline mt-4 inline-block'
            )

@ui.page('/chat')
def chat_ui():
    """Main chat interface"""
    global chat_history
    
    async def send_message():
        user_msg = input_box.value.strip()
        if not user_msg:
            return

        input_box.disable()
        send_btn.disable()

        chat_history.append(('user', user_msg))
        with chat_area:
            render_message('user', user_msg)
        
        input_box.value = ''
        
        with chat_area:
            loading_container = ui.row().classes('w-full justify-start mb-2')
            with loading_container:
                with ui.card().classes(f'bg-{'#3a5a5a' if dark_mode else '#f5d596'} text-{'#f8f8f8' if dark_mode else '#1a3a5f'} px-4 py-3 rounded-lg max-w-[80%]'):
                    with ui.row().classes('items-center gap-2'):
                        ui.spinner(size='sm')
                        ui.label('Searching authentic Islamic sources...').classes('text-sm')

        try:
            result = await process_islamic_query(user_msg)
            loading_container.delete()
            
            if result.get('success'):
                bot_response = {
                    'summary': result.get('summary', ''),
                    'quran': result.get('quran', ''),
                    'sahih_bukhari': result.get('sahih_bukhari', ''),
                    'sahih_muslim': result.get('sahih_muslim', '')
                }
                chat_history.append(('bot', bot_response))
                with chat_area:
                    render_detailed_response(bot_response)
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                chat_history.append(('bot', f"I apologize, but an error occurred: {error_msg}"))
                with chat_area:
                    render_message('bot', f"I apologize, but an error occurred: {error_msg}")
                
        except Exception as e:
            try:
                loading_container.delete()
            except:
                pass
            error_msg = "I apologize, but I'm having trouble processing your request right now. Please try again."
            chat_history.append(('bot', error_msg))
            with chat_area:
                render_message('bot', error_msg)
        
        finally:
            input_box.enable()
            send_btn.enable()
            input_box.focus()

    def render_message(sender, text):
        """Render a simple message bubble"""
        align = 'justify-end' if sender == 'user' else 'justify-start'
        bubble_color = 'bg-[#0e5449] text-white' if sender == 'user' else f'bg-{'#3a5a5a' if dark_mode else '#f5d596'} text-{'#f8f8f8' if dark_mode else '#1a3a5f'}'
        
        with ui.row().classes(f'w-full {align} mb-4'):
            with ui.card().classes(f'{bubble_color} px-4 py-3 rounded-lg max-w-[80%] shadow-sm'):
                ui.markdown(text).classes('text-sm leading-relaxed')

    def render_detailed_response(response_data):
        """Render detailed response with expandable sections"""
        with ui.row().classes('w-full justify-start mb-4'):
            with ui.card().classes(f'bg-{'#2d4a4a' if dark_mode else 'white'} border-l-4 border-{'#0e5449' if dark_mode else '#0e5449'} p-6 rounded-lg max-w-[90%] shadow-lg'):
                if response_data.get('summary'):
                    ui.label('📋 Unified Summary').classes(f'text-lg font-bold {'text-[#f5d596]' if dark_mode else 'text-[#0e5449]'} mb-3')
                    ui.markdown(response_data['summary']).classes(f'text-{'#d1d5db' if dark_mode else '#1a3a5f'} mb-6 leading-relaxed bg-{'#3a5a5a' if dark_mode else '#f8f9fa'} p-4 rounded-lg')
                
                sources = [
                    ('📖 Quran References', response_data.get('quran', ''), '#2d5a27'),
                    ('📚 Sahih Bukhari References', response_data.get('sahih_bukhari', ''), '#8b4513'),
                    ('📕 Sahih Muslim References', response_data.get('sahih_muslim', ''), '#4a5568')
                ]
                
                ui.label('Detailed References from Each Source:').classes(f'text-md font-semibold {'text-[#d1d5db]' if dark_mode else 'text-[#1a3a5f'} mb-3')
                
                for title, content, color in sources:
                    if content:
                        with ui.expansion(title).classes('w-full mb-2') as exp:
                            exp.props('dense').style(f'color: {color}; font-weight: 600;')
                            with exp:
                                ui.markdown(content).classes(f'text-{'#d1d5db' if dark_mode else '#1a3a5f'} leading-relaxed p-3 bg-{'#2d4a4a' if dark_mode else '#fafafa'} rounded')

    def render_chat_history():
        """Render the entire chat history"""
        with chat_area:
            for sender, message in chat_history:
                if sender == 'user':
                    render_message('user', message)
                elif isinstance(message, dict):
                    render_detailed_response(message)
                else:
                    render_message('bot', message)

    with ui.column().classes(f'h-screen w-full {'bg-[#1a2a2a]' if dark_mode else 'bg-gradient-to-b from-[#f8f8f8] to-[#e6d8c4]'} overflow-hidden m-0 p-0'):
        with ui.row().classes('fixed top-0 left-0 w-full justify-between items-center shadow-md z-10 h-16 px-4').style('background-color: #0e5449;'):
            with ui.row().classes('items-center gap-3'):
                ui.image('Logo1.png').classes('w-12 h-12')
                ui.label('DeenAI').classes('text-2xl font-bold text-[#f5d596]')
            
            with ui.row().classes('gap-2 items-center'):
                ui.link('🏠 Home', '/').classes('text-base text-[#f5d596] hover:text-white no-underline px-1 py-1 rounded transition-colors')
                ui.link('ℹ About', '/about').classes('text-base text-[#f5d596] hover:text-white no-underline px-3 py-1 rounded transition-colors')
                toggle_label = ui.label('🌙' if dark_mode else '☀').classes('text-2xl cursor-pointer ml-2')
                toggle_label.on('click', lambda: toggle_dark_mode())

        with ui.scroll_area().classes(
            'flex-1 w-full pt-20 pb-24 px-4 md:px-8 overflow-y-auto overflow-x-hidden'
        ) as chat_area:
            render_message('bot', '*Assalamu Alaikum wa Rahmatullahi wa Barakatuh!* 🌙\n\nWelcome to *DeenAI, your Islamic knowledge assistant. I\'m here to help you find authentic guidance from:\n\n• 📖 **The Holy Quran\n• 📚 **Sahih Bukhari\n• 📕 **Sahih Muslim*\n\nFeel free to ask me any questions about Islam, and I\'ll provide you with references from these authentic sources along with a unified summary.')
            render_chat_history()

        with ui.row().classes('fixed bottom-0 left-0 w-full px-6 py-3 z-10 justify-center items-center').style('background-color: #0e5449;'):
            def on_keydown(e):
                if e.args.get('key') == 'Enter':
                    if e.args.get('shiftKey'):
                        input_box.run_method('focus')
                        return
                    else:
                        asyncio.create_task(send_message())

            with ui.row().classes('w-full max-w-4xl items-center gap-2'):
                input_box = ui.input('Ask something about Islam...').props('dense').classes(
                    f'flex-grow px-4 py-3 border-2 border-{'#f5d596' if dark_mode else '#f5d596'} rounded-full {'bg-[#2d4a4a]' if dark_mode else 'bg-white'}'
                ).on('keydown', on_keydown)
                send_btn = ui.button('Send', on_click=lambda: asyncio.create_task(send_message())).classes(
                    f'bg-{'#2d4a4a' if dark_mode else '#f5d596'} hover:bg-{'#3a5a5a' if dark_mode else '#e6c580'} text-{'#f5d596' if dark_mode else '#020a37'} font-bold px-6 py-3 rounded-full transition-colors'
                ).props('no-caps').bind_visibility_from(input_box, 'enabled')

        input_box.on('mount', lambda: input_box.focus())

def update_placeholder(input_element):
    """Update the input placeholder based on content"""
    if not input_element.value.strip():
        input_element.set_value('Ask anything about Islam')
    elif input_element.value.startswith('Ask anything about Islam'):
        input_element.set_value(input_element.value.replace('Ask anything about Islam', '').strip())

def toggle_dark_mode():
    global dark_mode
    dark_mode = not dark_mode
    # Force immediate UI update for the current page
    if app.get_client().path in ['/about', '/chat']:
        ui.update()
    ui.notify(f'Switched to {"Dark" if dark_mode else "Light"} Mode')
    ui.update()


ui.run(title="DeenAI Islamic Assistant ")