"""Attack method implementations for image, INR, and embedding spaces."""
import json, os
import torch
from tqdm import tqdm
from basemodels.utils.nfn.nfn.common import WeightSpaceFeatures
from attacks.torchattacks.attacks.onepixel import OnePixel, BatchedOnePixelRunner

class INRPipelineWrapper(torch.nn.Module):
    def __init__(self, builder, victim, device='cuda'):
        super().__init__()
        self.builder = builder
        self.victim = victim
        self.victim_module = victim
        self._device = device
        self.init_params = None
        
    def forward(self, images):
        self.victim.eval()
        with torch.no_grad():
            batch_size = images.shape[0]
            weights, biases = self.init_params

            if weights[0].shape[0] != batch_size:
                init = (
                    tuple(w.expand(batch_size, *w.shape[1:]) for w in weights),
                    tuple(b.expand(batch_size, *b.shape[1:]) for b in biases)
                )
            else:
                init = self.init_params

            wts_and_bs = self.builder.build_wts_and_bs_from_img_fast(
                images, WeightSpaceFeatures(*init)
            )
            inp = WeightSpaceFeatures(*wts_and_bs).to(self._device)
            logits = self.victim(inp)
        return logits

# run_onepixel.py에 subset 인자 추가
def run_onepixel(
        dataset_name,
        device='cuda', 
        save_intermediate_results=False,
        **kwargs):

    victim = kwargs["victim_wrapped"]
    victim.eval()

    _, _, testloader = kwargs['dataloader']
    builder = kwargs['builder']
    
    # GPU별 subset 분배
    # gpu_id = kwargs.get('gpu_id', 0)
    # num_gpus = kwargs.get('num_gpus', 1)
    pipeline = INRPipelineWrapper(builder, victim, device=device)
    runner = BatchedOnePixelRunner(
        pipeline, 
        pixels=kwargs.get("pixels", 1), 
        steps=kwargs.get("steps", 10), 
        popsize=kwargs.get("popsize", 10)
    )
    print(f"Initialized OnePixel attack with pixels={runner.pixels}, steps={runner.steps}, popsize={runner.popsize}, batch_size={kwargs.get('batch_size', 4)}")
    print(f"attack_inr_fitting_steps: {kwargs.get('attack_inr_fitting_steps', 500)}, attack_inr_fitting_lr: {kwargs.get('attack_inr_fitting_lr', 5.0e-4)}")
    pipeline.eval()

    clean_correct, adv_correct, total = 0, 0, 0
    max_samples = kwargs.get('max_samples', 1000)
    start_batch = kwargs.get("start_batch", 0)

    for i, batch in tqdm(enumerate(testloader)):
        if total >= max_samples:
            break
        if i < start_batch:
            continue
            
        inrs, init_params, imgs, labels = batch
        labels = labels.to(device, non_blocking=True)
        srcs = imgs.to(device, non_blocking=True)
        builder.steps = kwargs.get("attack_inr_fitting_steps", 500)
        builder.lr = kwargs.get("attack_inr_fitting_lr", 5.0e-4)

        adv_srcs = runner.attack_batch(srcs, labels, init_params)
        builder.steps = 5000
        builder.lr = 5.0e-04

        pipeline.init_params = init_params
        with torch.no_grad():
            # clean_pred = pipeline(srcs).argmax(dim=1)
            # clean_correct += (clean_pred == labels).sum().item()

            adv_pred = pipeline(adv_srcs).argmax(dim=1)
            adv_correct += (adv_pred == labels).sum().item()

            total += labels.numel()

        clean_correct = -1

        # print(f"BATCH {i} | clean acc: {clean_correct / max(total, 1):.4f} ({clean_correct}/{total})", flush=True)
        print(f"BATCH {i} | Adv acc: {adv_correct / max(total, 1):.4f} ({adv_correct}/{total})", flush=True)

    result = {"clean_correct": clean_correct, "adv_correct": adv_correct, "total": total}
    # 결과 저장
    # save_path = os.path.join(kwargs.get('exp_path', '.'), f"onepixel_gpu{gpu_id}.json")
    # with open(save_path, 'w') as f:
    #     json.dump(result, f)

    return result
