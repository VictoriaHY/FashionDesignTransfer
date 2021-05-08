## Network

<div align="center">
<img src="https://github.com/VictoriaHY/MM2020/blob/4d441fbef1f7b065190502680158408376b752a5/pics/framework2label.jpg" height="160px">
</div>

## Usage

```python
python xxx.py --dataroot ./data --name xxx --model model_name --direction BtoA(or B2A)
```

train.py: training G_d and G_s

train2.py: traiining the two-step network after G_d (fix G_d)

train3.py: training fusion network for G_d and G_s (fix G_d and G_s)

test.py: test G_d and G_s

test1.py: test the fusion network of G_d and G_s

