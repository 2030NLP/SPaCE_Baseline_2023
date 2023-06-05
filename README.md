# SPaCE2023 Baseline

## 安装
从GitHub上clone该项目后，执行：
```bash
cd src/
pip install -r requirement.txt
cd ..
```
后续操作都在主文件夹（`data, scripts, src`的上级文件夹）下执行。

## Task1
Task1的基线模型使用序列标注的方式解决该任务。
模型为每一个token预测0~6的标签，1-6分别代表`S1 P1 E1 S2 P2 E2`，0代表一般文本。

运行baseline的步骤为：
```bash
sh scripts/task1/task1_train.sh
sh scripts/task1/task1_predict.sh
sh scripts/task1/task1_scoring.sh
```
三条指令分别对应训练、预测、评分步骤。

## Task2
Task2的基线模型使用事件抽取的方式解决该任务。
模型首先抽取文本中所有潜在的“事件”元素，之后以每个“事件”元素为核心，抽取文本中相关的其他元素组成元组。

运行baseline的步骤为：
```bash
sh scripts/task2/task2_train_trigger.sh
sh scripts/task2/task2_train_element.sh
sh scripts/task2/task2_predict.sh
sh scripts/task2/task2_scoring.sh
```
四条指令分别对应训练抽取事件模型、训练抽取其余元素模型、预测、评分步骤。