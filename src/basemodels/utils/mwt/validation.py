import os
import time
import torch
from torch.amp import GradScaler, autocast
from .datasets import unpack

def validation(loader_val, 
                name, 
                forward, 
                device, 
                PI, 
                siren_rgb, 
                classifier, 
                meta_sgd, 
                val_data, 
                conf):
    all_predicted_labels = []
    all_actual_labels = []
    all_psnrs = []

    with autocast(enabled=conf['auto_cast'], device_type=device, dtype=torch.float16):
        timings = []
        inference_start = time.time()
        time_since_print = 0

        ijk = 0
        for tp in loader_val:
            ijk += 1
            image, label = unpack(tp, conf, device)
            logits, _, psnrs = forward(conf, 
                                        image, None, 
                                        build_graph=False, 
                                        device=device, 
                                        PI=PI, 
                                        siren_rgb=siren_rgb, 
                                        classifier=classifier, 
                                        meta_sgd=meta_sgd)
            pred = logits.argmax(dim=1) # [N]
            
            all_predicted_labels.append(pred.detach())
            all_actual_labels.append(label.detach())
            all_psnrs.append(psnrs.detach())

            # if (time.time() - time_since_print) > 10:
            #     time_since_print = time.time()
            #     # print('validation scale', image.min(), image.max())
            #     print('so far list sizes', len(all_predicted_labels), len(all_actual_labels), len(all_psnrs))
            #     print('pred', pred)
            #     print('actual', label)
            #     print('acc of batch',  (pred == label).long().sum().item() / pred.shape[0])
            #     print('time taken so far', (time.time() - inference_start) / 60.0, 'minutes', flush=True)
            #     print()

            assert len(all_predicted_labels) == len(all_actual_labels) == len(all_psnrs)
        
        torch.cuda.synchronize()
        inference_end = time.time()
        val_took = (inference_end - inference_start)

        all_actual_labels = torch.cat(all_actual_labels)
        all_predicted_labels = torch.cat(all_predicted_labels)

        # checks
        assert all_actual_labels.shape == all_predicted_labels.shape
        assert len(all_actual_labels.shape) == 1
        assert all_actual_labels.shape[0] == len(loader_val.dataset)

        correct_count = (all_actual_labels == all_predicted_labels).long().sum().item()
        total_count = all_actual_labels.shape[0]
        acc = correct_count / total_count
        psnr_avg = torch.cat(all_psnrs).mean()
        print(f'Validation {name} accuracy: {acc}, psnr: {psnr_avg} computed from {correct_count} / {total_count} samples')

        val_data[f'acc_{name}'] = float(acc)
        val_data[f'psnr_{name}'] = float(psnr_avg)

        # also compute acc per gt class
        acc_per_class = []
        label_count = torch.amax(all_actual_labels).item() + 1
        for c in range(label_count):
            mask = (all_actual_labels == c)
            correct_count = (all_actual_labels[mask] == all_predicted_labels[mask]).long().sum().item()
            total_count = mask.long().sum().item()
            pair = ('c=' + str(c), correct_count, total_count, str(round(float(100 * correct_count / (total_count + 1e-6)), 2)) + '%')
            acc_per_class.append(pair)
        print('Accuracies per class', acc_per_class)
