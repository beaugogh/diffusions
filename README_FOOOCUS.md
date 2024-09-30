## Fooocus tips

#### Upscale Images without Distorting Face
* Option 1: use `Upscale (Fast 2x)` option
* Option 2: Go to `Advanced`, enable `Developer Debug Mode`, go to `Debug  Tools` tab, change the value of the option `Forced Overwrite of Denoising Strength of "Upscale"` from -1 to 0.01.
* Reference: https://www.youtube.com/watch?v=Tph6T0bgafE 
  
#### Variation with More Control
  1. Mix variation with image prompt: Go to `Advanced`, enable `Developer Debug Mode`, go `Control` tab, check `Mixing Image Prompt and Vary/Upscale`.
  2. Fine-grained control of the variation parameter: `Advanced` > `Developer Debug Mode` > `Debug  Tools` > `Forced Overwrite of Denoising Strength of "Vary"`. `-1` means disabled, `0.5` means subtle variation, `0.85` means strong variation. (see https://www.youtube.com/watch?v=9yDwJe5ddfM)

#### LoRAs
* Gender Transition Slider
  * link: https://civitai.com/models/466010/gender-transition-slider
  * base model: Pony
  * usage: `<lora:Gender Swap_alpha1.0_rank4_noxattn_last:-0.5>` male, `<lora:Gender Swap_alpha1.0_rank4_noxattn_last:0>` female
* Age Slider LoRA | PonyXL SDXL
  * link: https://civitai.com/models/402667/age-slider-lora-or-ponyxl-sdxl
  * base model: Pony
  * prompt usage: `NO_LORA` no change, `<lora:age_slider_v4:-5>` very young, `<lora:age_slider_v4:5>` very mature


#### Checkpoints
* Pony
  * link: https://civitai.com/models/257749/pony-diffusion-v6-xl
  * setting: `Advanced` > `Guidance Scale` > 7
  * prompt usage: 
    * `score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, just describe what you want, tag1, tag2`
    * `score_9, score_8_up, score_7_up, score_6_up, just describe what you want`
