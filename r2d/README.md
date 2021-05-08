## Network

<div align="center">
<img src="https://github.com/VictoriaHY/MM2020/blob/4d441fbef1f7b065190502680158408376b752a5/pics/framework2label.jpg" height="160px">
</div>

## Usage

```python
python xxx.py --dataroot ./data --name xxx --model model_name --direction BtoA(or B2A)
```

train.py: trainnig G_a

train2.py: training the fusion network after w/o label G_a​ (fix G_d)

train3.py: training the fusion network after G_a​ (fix G_a)

test.py: test G_a

test1.py: test fusion network

