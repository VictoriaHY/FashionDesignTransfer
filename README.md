## Fashion Design Transfer

This is the official PyTorch implementation of the paper [From Design Draft to Real Attire: Unaligned Fashion Image Translation](https://victoriahy.github.io/MM2020/) by: 

<div align="center">
<img src="https://github.com/VictoriaHY/MM2020/blob/e6137ff65d971bd4316a74cc71285841e0f82b6e/pics/authors.jpg" height="160px">
</div>

This paper allows translations between unaligned design drafts and real fashion items. Compared with Pix2pix, our method generates accurate shapes and preserves vivid texture details:

<div align="center">
<img src="https://github.com/VictoriaHY/MM2020/blob/5633f4b26f86f3fd1bd5d99fa7ddde6e706a3c2b/pics/teaser1.jpg" height="160px">
</div>

Also, it has application in fashion design editing. Top row: input design draft and rendered fashion items by different methods. Bottom row: edited design draft and the corresponding modified fashion items:

<div align="center">
<img src="https://github.com/VictoriaHY/MM2020/blob/5633f4b26f86f3fd1bd5d99fa7ddde6e706a3c2b/pics/application.jpg" height="160px">
</div>

## Usage

Please check both d2r and r2d for how to use them.

## Acknowledgments

This code borrows heavily from [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) . Please follow [dataset for pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md) to create your own dataset.

## Citation

If you want to cite our research, please use:

```
@inproceedings{han2020design,
  title={From Design Draft to Real Attire: Unaligned Fashion Image Translation},
  author={Han, Yu and Yang, Shuai and Wang, Wenjing and Liu, Jiaying},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={1533--1541},
  year={2020}
}
```

