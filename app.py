import json
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(
    __name__,
    static_folder='app/static',
    template_folder='app/templates'
)

MODEL_NAME = "openai-community/gpt2"

class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_weights = []

    def extract_attention(self, input_ids, tokenizer):
        self.attention_weights = []

        def hook(module, inp, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1]
                self.attention_weights.append(attn.detach().cpu().numpy())

        hooks = [layer.attn.register_forward_hook(hook) for layer in self.model.transformer.h]

        with torch.no_grad():
            _ = self.model(input_ids, output_attentions=True)

        for h in hooks:
            h.remove()

        token_texts = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        formatted = {"tokens": [], "layers": []}
        for i, t in enumerate(token_texts):
            formatted["tokens"].append({"text": t, "index": i})

        for layer_idx, layer_w in enumerate(self.attention_weights):
            layer_w = layer_w[0]
            layer_entry = {"index": layer_idx, "heads": []}
            for head_idx, head_w in enumerate(layer_w):
                sparse = []
                seq_len = head_w.shape[0]
                for i in range(seq_len):
                    for j in range(seq_len):
                        v = float(head_w[i, j])
                        if v > 0.01:
                            sparse.append([i, j, v])
                layer_entry["heads"].append({"index": head_idx, "weights": sparse})
            formatted["layers"].append(layer_entry)

        return formatted

# Globals
tokenizer = None
model = None
extractor = None

def load_model():
    global tokenizer, model, extractor
    if tokenizer is None:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True)
        extractor = AttentionExtractor(model)
        print("Model and tokenizer loaded!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/attention', methods=['POST'])
def analyze_attention():
    data = request.json or {}
    input_text = data.get('text', '')
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        load_model()
        inputs = tokenizer(input_text, return_tensors="pt")
        if inputs.input_ids.shape[1] > 100:
            return jsonify({"error": "Text too long. Please limit to 100 tokens."}), 400

        attention_data = extractor.extract_attention(inputs.input_ids, tokenizer)
        return jsonify(attention_data)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)