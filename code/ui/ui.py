import gradio as gr
from app import ask_question, get_answer, select_retriever, select_chunking_strategy 

def update_visibility(image):
    return {"visible": image is not None}

def handle_output(markdown1, image, markdown2):
    visible = image is not None
    if markdown2 is None:
        markdown2_update = gr.update(visible=False)
    else:
        markdown2_update = gr.update(value=f"**Formula:**\n{markdown2}", visible=True)

    markdown1_update = gr.update(value=f"**Textual answer:**\n\n\n{markdown1}", visible=True)
    return markdown1_update, image, markdown2_update, gr.update(visible=visible)

def show_options(choice):
  if choice == "RAG with ISLP.pdf parsed by unstructured.io +  Multivector Retriever ":
    return gr.update(visible = True)
  else:
    return gr.update(visible=False)


CSS = """
.my-button {
    background-color: #C3D6A4;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.my-button:hover {
    background-color: #A8B77B;
}

.my-button:active {
    background-color: #8D9B6A;
}


"""

def txt_answer_visibility_fn(question,method):
      answer_txt = get_answer(question, method)
      return answer_txt, gr.update(visible=True) # odgovor, vidljivost txt boxa za odgovor, vidljivost drugog txt boxa za odg chat-a

def create_ui():
  with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="lime")) as ui:   # gr.themes.Monochrome(), gr.themes.Soft(primary_hue="lime")
      gr.Markdown("<h1 style='text-align: center;'>RAG Chatbot with Introduction to statistical learning with Python textbook! 🤖</h1>")


      with gr.Row():
          with gr.Column():
              method_dropdown = gr.Dropdown(
                  label="Choose Query Translation Type:",
                  choices=["Multiquery Chain", "Recursive Decomposition", "Query Decomposition (Individual Answers)", "Step Back Chain", "HyDE Chain"],
                  value="Multiquery Chain"
              )
              radio = gr.Radio(
                  choices=["RAG with ISLP.mmd + vectostore as retreiver", "RAG with ISLP.pdf parsed by unstructured.io +  Multivector Retriever "], 
                  label="Choose an option:"
              )
              
              radio2 = gr.Radio(choices=["Chunking strategy 1", "Chunking strategy 2", "Chunking strategy 3", "Chunking strategy 4"], label="Choose chunking strategy.",visible=False)
              radio.change(select_retriever, inputs=radio)


              radio2.change(select_chunking_strategy,inputs=radio2)

              radio.change(show_options,radio,radio2)

              question_box_for_qt = gr.Textbox(label="Try different Query Translation methods! (Only Textual output expected)", placeholder="Enter your message...")
              submit_btn_for_qt = gr.Button("Get Answer using selected Query Translation Method and retriever", elem_classes="my-button")
              answer_output = gr.Markdown(label="Answer:", visible=False)   # bio je Textbox

              markdown_output1 = gr.Markdown()
              image_output = gr.Image(label="Figure", interactive=False, visible=False)
              markdown_output2 = gr.Markdown(label="Formula")

              submit_btn_for_qt.click(lambda _:gr.update(visible=False), None, image_output).then(lambda _:gr.update(visible=False),None,markdown_output1).then(txt_answer_visibility_fn, inputs=[question_box_for_qt, method_dropdown], outputs=[answer_output,answer_output])

          with gr.Column():
              question_box = gr.Textbox(label="Chatbot (uses multi-query processing and fine-tuned ColBERT models for recognizing images and formulas)", placeholder="Enter your message...")
              submit_btn = gr.Button("Ask question", elem_classes="my-button")
              markdown_output1 = gr.Markdown()
              submit_btn.click(lambda _: gr.update(visible=False), None, answer_output).then(ask_question, inputs=[question_box], outputs=[markdown_output1, image_output, markdown_output2]).then(handle_output, inputs=[markdown_output1, image_output, markdown_output2], outputs=[markdown_output1, image_output, markdown_output2, image_output])


      with gr.Row():
          clear_txt_search_btn = gr.Button("Clear text search bar")
          reset_txt_results_btn = gr.Button("Reset textual results")
          #undo_btn = gr.Button("Undo")
          clear_btn = gr.Button("Clear chatbot search bar")
          reset_interface_btn = gr.Button("Reset chatbot retrieval results")

          clear_btn.click(lambda: "", None, question_box)
          reset_interface_btn.click(
              lambda: ["", gr.update(value=None, visible=False), ""],
              None,
              [markdown_output1, image_output, markdown_output2]
          )
          clear_txt_search_btn.click(lambda: "", None, question_box_for_qt)
          reset_txt_results_btn.click(
              lambda: gr.update(value="",visible=False),
              None,
              answer_output
          )
  return ui.launch()