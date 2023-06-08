from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
import gradio as gr


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


def question_answer(image, questions):
    questions = [line.strip() for line in questions.strip().splitlines()]

    answers = []

    for question in questions:
        with torch.no_grad():
            # prepare inputs
            encoding = processor(image, question, return_tensors="pt")
            # forward pass
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answers.append(
                "Question: " + question + "\nAnswer: " + model.config.id2label[idx]
            )

    return "\n\n".join(answers)


iface = gr.Interface(
    fn=question_answer,
    inputs=["image", "textarea"],
    outputs=["textarea"],
)
iface.launch()
