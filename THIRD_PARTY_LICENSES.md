# Third-Party Licenses

This package redistributes and/or depends on third-party components.

## 1) BiRefNet / ToonOut (code + weights)

- Upstream: https://github.com/MatteoKartoon/BiRefNet
- License: MIT
- Copyright:
  - Original BiRefNet: Copyright (c) 2024 ZhengPeng
  - ToonOut modifications: Copyright (c) 2025 Matteo Muratori, Joël Seytre
- Full text included in: `LICENSE.BiRefNet-ToonOut`

### Recommended citation

- Muratori, Matteo and Seytre, Joël.
- "ToonOut: Fine-tuned Background Removal for Anime Characters"
- arXiv: https://arxiv.org/abs/2509.06839

## 2) ToonOut Dataset (only if redistributed)

- License: CC BY 4.0
- Attribution required: Matteo Muratori, Joël Seytre
- Full text included in: `LICENSE.ToonOut-Dataset.CC-BY-4.0`
- License URL: https://creativecommons.org/licenses/by/4.0/

If you do not redistribute the dataset, this section is informational only.

## Redistribution checklist

- Keep `LICENSE.BiRefNet-ToonOut` in distributed artifacts.
- Keep `NOTICE.txt` and this file in distributed artifacts.
- If dataset is redistributed, include `LICENSE.ToonOut-Dataset.CC-BY-4.0` and attribution.
