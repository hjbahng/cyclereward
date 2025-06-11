import os
import argparse
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    set_seed,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
from diffusers import (
    DiffusionPipeline, 
    StableDiffusion3Pipeline,
    FluxPipeline, 
    StableDiffusionPipeline
)
from vllm import LLM, SamplingParams
from dreamsim import dreamsim
from sentence_transformers import SentenceTransformer


def run_llava(question):
    q = "Write a detailed description of the given image." if question == "default" else question
    prompt = f"USER: <image>\n{q}\nASSISTANT:"
    return prompt, None

def run_llava_next(question):
    q = "Write a detailed description of the given image." if question == "default" else question
    prompt = f"[INST] <image>\n{q} [/INST]"
    return prompt, None

def run_llava_34b(question):
    q = "Write a detailed description of the given image." if question == "default" else question
    prompt = f"<|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n"
    return prompt, None

def run_llava_onevision(question):
    q = "Write a detailed description of the given image." if question == "default" else question
    prompt = f"<|im_start|>user <image>\n{q}<|im_end|> \
    <|im_start|>assistant\n"
    return prompt, None

def run_blip2(question):
    prompt = "this is a picture of" if question == "default" else question
    return prompt, None

def run_internvl(question):
    q = "Please describe the image in detail." if question == "default" else question
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-2B", trust_remote_code=True)
    messages = [{'role': 'user', 'content': f"<image>\n{q}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return prompt, stop_token_ids

model_example_map = {
    "llava-hf/llava-1.5-7b-hf": run_llava,
    "llava-hf/llava-1.5-13b-hf": run_llava,
    "llava-hf/llava-v1.6-mistral-7b-hf": run_llava_next,
    "llava-hf/llava-v1.6-vicuna-7b-hf": run_llava_next,
    "llava-hf/llava-v1.6-34b-hf": run_llava_34b,
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": run_llava_onevision,
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": run_llava_onevision,
    "Salesforce/blip2-opt-2.7b": run_blip2,
    "Salesforce/blip2-opt-6.7b": run_blip2,
    "Salesforce/blip2-flan-t5-xxl": run_blip2,
    "OpenGVLab/InternVL2-2B": run_internvl,
    "OpenGVLab/InternVL2-8B": run_internvl,
    "OpenGVLab/InternVL2-26B": run_internvl,
    "OpenGVLab/InternVL2-40B": run_internvl
}

def run_vllm(model, image, args):
    prompt_fn = model_example_map[args.model_name_or_path]
    prompt, stop_token_ids = prompt_fn(args.question)
    
    if args.model_name_or_path == "Salesforce/blip2-flan-t5-xxl":
        processor = Blip2Processor.from_pretrained(args.model_name_or_path)
        inputs = processor(image, prompt, return_tensors="pt").to("cuda", torch.float16)
        outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, temperature=args.temperature, do_sample=False)
        return processor.decode(outputs[0], skip_special_tokens=True).strip()
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids
    )
    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
    outputs = model.generate(inputs, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()

@torch.no_grad()
def generate_captions(dataset, model, args, result_path, reconstruct=False):
    results = []
    for example in tqdm(dataset, desc="Generating captions"):
        image_file = example['fake_image'] if reconstruct else example['real_image']
        image = Image.open(image_file).convert('RGB')
        caption = run_vllm(model, image, args)
        if reconstruct:
            data_dict = {**example, "recon_text": caption}
        else:
            data_dict = {**example, "fake_text": caption}
        results.append(data_dict)
        
    json.dump(results, open(result_path, 'w'))
    print("Saved results to: ", result_path)
    return results

@torch.no_grad()
def generate_images(dataset, pipe, generator, args, image_path, result_path, reconstruct=False):
    results = []
    for i, example in enumerate(tqdm(dataset, desc="Generating images")):
        text = example["fake_text"] if reconstruct else example["real_text"]
        image = pipe(prompt=text,
                    generator=generator, 
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.resolution,
                    width=args.resolution).images[0]
        image_file = os.path.join(image_path, f"gen_{i}.png")
        image.save(image_file)
        if reconstruct:
            data_dict = {**example, "recon_image": image_file}
        else:
            data_dict = {**example, "fake_image": image_file}
        results.append(data_dict)
        
    json.dump(results, open(result_path, 'w'))
    print("Saved results to: ", result_path)
    return results

def get_metrics_image(results, ds_preprocess, ds_model, device, metrics_path):
    """Measure similarity between real and reconstructed images."""
    metrics = []
    for result in tqdm(results):
        real_image_file = result["real_image"]
        recon_image_file = result["recon_image"]
        real_image = Image.open(real_image_file).convert("RGB")
        recon_image = Image.open(recon_image_file).convert("RGB")

        # measure image similiarity
        img1 = ds_preprocess(real_image).to(device)
        img2 = ds_preprocess(recon_image).to(device)
        dreamsim_score = ds_model(img1, img2).float().item()

        # save results
        metrics.append({**result, "dreamsim": dreamsim_score})
    
    json.dump(metrics, open(metrics_path, 'w'))
    print("Saved metrics to: ", metrics_path)
    return metrics

def get_metrics_text(results, sbert_scorer, criterion, metrics_path):
    """Measure similarity between real and reconstructed texts."""
    metrics = []
    for result in tqdm(results):
        real_text = result["real_text"]
        recon_text = result["recon_text"]

        # measure text similiarity
        embeddings = sbert_scorer.encode([real_text, recon_text])
        embeddings = torch.from_numpy(embeddings)
        sbert_score = criterion(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).float().item()

        # save results
        metrics.append({**result, "sbert_score": sbert_score})
    
    json.dump(metrics, open(metrics_path, 'w'))
    print("Saved metrics to: ", metrics_path)
    return metrics

def load_F_model(args):
    if args.model_name_or_path == "Salesforce/blip2-flan-t5-xxl":
        return Blip2ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    kwargs = dict(
        model=args.model_name_or_path, 
        dtype=torch.bfloat16,
        download_dir=args.cache_dir, 
    )
    if "mistral" in args.model_name_or_path:
        kwargs["max_model_len"] = 8192
    elif "llava-v1.6-34b-hf" in args.model_name_or_path:
        kwargs["max_model_len"] = 4096
    elif "onevision" in args.model_name_or_path:
        kwargs["max_model_len"] = 32768
    return LLM(**kwargs)

def load_G_model(args):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-3-medium-diffusers":
        print("Loading stable diffusion 3...")
        torch.set_float32_matmul_precision("high")

        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir
        )
        pipe.set_progress_bar_config(disable=True)

        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
        pipe.to(f'cuda:{local_rank}')
        args.resolution = 1024

    elif args.pretrained_model_name_or_path == "sd-legacy/stable-diffusion-v1-5":
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, 
            torch_dtype=torch.float16, 
            download_dir=args.cache_dir
        )
        pipe.to(f'cuda:{local_rank}')
        args.resolution = 512

    elif args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-xl-base-1.0": 
        pipe = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            variant='fp16',
            cache_dir=args.cache_dir,
        )
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.to(f'cuda:{local_rank}')
        args.resolution = 1024

    elif args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-schnell": 
        pipe = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            download_dir=args.cache_dir)
        pipe.to(f'cuda:{local_rank}')
        args.num_inference_steps = 4
        args.guidance_scale = 0
        args.resolution = 1024

    generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(args.seed) if args.seed else None
    return pipe, generator

def load_dci(data_path):
    ann_path = os.path.join(data_path, "annotations")
    img_path = os.path.join(data_path, "photos")
    split_path = os.path.join(data_path, "splits.json")
    split_metadata = json.load(open(split_path, 'r'))
    sources = split_metadata['train']

    image_dataset = []
    text_dataset = []
    for source_file in tqdm(sources, desc="Load DCI dataset"):
        file_path = os.path.join(ann_path, source_file)
        base_data = json.load(open(file_path, 'r'))
        img_file = os.path.join(img_path, base_data['image'])
        image_dataset.append({"real_image": img_file})
        text_dataset.append({"real_text": base_data['short_caption']})
    return image_dataset, text_dataset

def create_save_path(args):
    f_model = args.model_name_or_path.split("/")[-1]
    g_name = args.pretrained_model_name_or_path.split("/")[-1]
    g_model = f"{g_name}-{args.seed}"

    forward_model = f_model if args.cycle == "i2t2i" else g_model
    forward_dir = os.path.join(args.output_path, args.dataset, "generated_data", forward_model)
    forward_path = os.path.join(forward_dir, "results.json")
    os.makedirs(forward_dir, exist_ok=True)

    model_combo = f"{f_model}_{g_model}"
    backward_dir = os.path.join(args.output_path, args.dataset, model_combo, args.cycle)
    backward_path = os.path.join(backward_dir, "results.json")
    os.makedirs(backward_dir, exist_ok=True)

    image_dir = backward_dir if args.cycle == "i2t2i" else forward_dir
    image_path = os.path.join(image_dir, "images")
    os.makedirs(image_path, exist_ok=True)

    metrics_path = os.path.join(backward_dir, 'metrics.json')
    return forward_path, backward_path, image_path, metrics_path

def main():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--cycle', type=str, default='i2t2i', choices=['i2t2i', 't2i2t'], help='Which cycle to run?')
    parser.add_argument('--dataset', type=str, default='DCI', help='Source of input data')
    parser.add_argument('--data_path', type=str, default="densely_captioned_images", help='Data path for DCI')
    parser.add_argument('--output_path', type=str, default="results", help='Path to save results')
    parser.add_argument('--cache_dir', type=str, default="~/.cache/huggingface/hub", help='Cache directory')
    parser.add_argument('--seed', type=int, default=123, help='Random seed') 

    # Image-to-text model arguments
    parser.add_argument('--model_name_or_path', type=str, default="llava-hf/llava-1.5-13b-hf", 
                        help='Name of the image-to-text model')
    parser.add_argument('--question', type=str, default="default", help='Prompt for generating captions')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature sampling')
    parser.add_argument('--max_tokens', type=float, default=77, help='Maximum tokens to generate')
    
    # Text-to-image model arguments
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", 
                        help='Name of the text-to-image model')
    parser.add_argument('--num_inference_steps', type=float, default=50, 
                        help='The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale for classifier-free guidance. Higher values lead to more faithful images but less diversity.')
    parser.add_argument('--resolution', type=int, default=512, help='Image height and width')
    parser.add_argument('--variant', type=str, default="fp16", help='Model variant')

    args = parser.parse_args()
    set_seed(args.seed)

    # create save paths
    forward_path, backward_path, image_path, metrics_path = create_save_path(args) 
    
    # we provide DCI as an example, but you can use 
    # any unpaired texts or images in the following format:
    # image_dataset = [{'real_image': image_path}, ...]
    # text_dataset = [{'real_text': caption}, ...]
    image_dataset, text_dataset = load_dci(args.data_path)
    
    if args.cycle == "i2t2i":
        # forward cycle
        if os.path.exists(forward_path):
            print("Loading existing dataset")
            forward_results = json.load(open(forward_path, 'r'))
        else:
            model = load_F_model(args)
            forward_results = generate_captions(image_dataset, model, args, forward_path)
            del model
            torch.cuda.empty_cache()
        
        # backward cycle
        pipe, generator = load_G_model(args)
        backward_results = generate_images(forward_results, pipe, generator, args, image_path, backward_path, reconstruct=True)
        
        del pipe, generator
        torch.cuda.empty_cache()

        # compute cycle consistency score
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ds_model, ds_preprocess = dreamsim(pretrained=True) # lower is similar
        get_metrics_image(backward_results, ds_preprocess, ds_model, device, metrics_path)   

    elif args.cycle == 't2i2t':
        # forward cycle
        if os.path.exists(forward_path):
            print("Loading existing dataset")
            forward_results = json.load(open(forward_path, 'r'))
        else:
            pipe, generator = load_G_model(args)
            forward_results = generate_images(text_dataset, pipe, generator, args, image_path, forward_path)
            del pipe
            torch.cuda.empty_cache()

        # backward cycle
        print(forward_results[0])
        model = load_F_model(args)
        backward_results = generate_captions(forward_results, model, args, backward_path, reconstruct=True)

        del model
        torch.cuda.empty_cache()

        # compute cycle consistency score
        sbert_scorer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # higher is similar
        criterion = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        get_metrics_text(backward_results, sbert_scorer, criterion, metrics_path)
    
if __name__ == '__main__':
    main()