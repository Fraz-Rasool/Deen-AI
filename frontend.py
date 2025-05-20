# import nicegui
from nicegui import ui

# === OPTIONAL: Theme Toggle (Dark/Light) ===
with ui.row().classes('absolute top-4 right-4 z-10'):
    ui.toggle(['🌞', '🌙'], value='🌙').bind_value(ui.dark_mode).classes('text-sm')


# === ROUTE 1: LANDING PAGE ===
@ui.page('/')
def landing():
    # Background video
    with ui.element('div').classes('fixed inset-0 -z-10 overflow-hidden'):
        ui.video('kabba.mp4', autoplay=True, muted=True, loop=True).classes('w-full h-full object-cover')

    # Overlay Welcome Content
    with ui.column().classes('absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center text-white'):
        ui.label('Welcome to Deen AI').classes('text-5xl font-bold mb-6')
        ui.label('Ask anything about Islam, powered by authentic sources.').classes('text-xl mb-10')
        ui.link('Enter Chat', '/chat').classes('bg-white text-black px-6 py-3 rounded-lg shadow-lg text-lg')



# === ROUTE 2: CHAT PAGE ===
from nicegui import ui

@ui.page('/chat')
def chat_page():
    chat_history = []

    def send_message():
        user_msg = input_box.value.strip()
        if not user_msg:
            return

        chat_history.append(('user', user_msg))
        with messages_container:
            render_message('user', user_msg)

        # Replace this with your real model response
        bot_reply = "This is where your model response goes."
        chat_history.append(('bot', bot_reply))
        with messages_container:
            render_message('bot', bot_reply)

        input_box.value = ''

    def render_message(sender, text):
        if sender == 'user':
            with ui.row().classes('justify-end w-full'):
                ui.label(text).classes(
                    'bg-[#3B2F2F] text-white px-4 py-2 rounded-xl max-w-[70%] text-right'
                )
        else:
            with ui.row().classes('justify-start w-full'):
                ui.label(text).classes(
                    'bg-gray-200 text-black px-4 py-2 rounded-xl max-w-[70%] text-left'
                )

    # === Page Background Wrapper ===
    with ui.column().classes('min-h-screen w-full p-6 bg-gray-700'):

        # === Header Section ===
        with ui.row().classes('items-center gap-3 mb-6 p-3 rounded-xl bg-[#3B2F2F] shadow-lg'):
            ui.image('logo (2).png').classes('w-10 h-10')
            ui.label('Deen AI Chatbot').classes('text-xl font-bold text-white')

        # === Chat Area ===
        with ui.column().style(
            '''
            height: 500px;
            width: 900px;
            background-color: white;
            border-radius: 20px;
            padding: 20px;
            margin-right: 200px;
            margin-left: 250px;

            display: flex;
            flex-direction: column;
            '''
        ) as chat_column:

            # Scrollable chat messages container - grows to fill space
            with ui.column().style(
                '''
                flex-grow: 1;
                overflow-y: auto;
                margin-bottom: 1rem;
                width: 850px
                '''
            ) as messages_container:
                # Initial bot greeting
                render_message('bot', 'Assalamu Alaikum! Ask me anything about Islam.')

            # Input and send button fixed at bottom
            with ui.row().classes('mt-4').style(
                '''
                width: 100%;
                gap: 0.5rem;
                align-items: center;
                '''
            ):
                def on_keydown(e):
                    key = e.args.get('key')
                    if key == 'Enter':
                        send_message()
                        e.prevent_default()

                input_box = ui.input('Ask something about Islam...').style(
                    '''
                    flex-grow: 1;
                    border-radius: 10px;
                    border: 1px solid #ccc;
                    padding: 0.5rem 1rem;
                    '''
                ).on('keydown', on_keydown)

                ui.button('SEND', on_click=send_message).classes(
                    'bg-[#3B2F2F] text-white px-6 py-2 rounded hover:bg-[#2e2323]'
                ).style('border-radius: 10px;')




# === RUN ===
ui.run(title='Deen AI Chatbot')
