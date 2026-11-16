# 测试集数据集

该目录包含一个小型测试集，内含5道问题以及对应的企业报告。
你可以借助它来研读示例问题、报告与系统输出结果。

## 内容清单

- `questions.json`：包含5道测试问题
- `subset.csv`：测试文档的元数据
- `pdf_reports/`：存放原始PDF报告的目录
- `answers_max_nst_o3m.json`：优胜系统给出的样例答案
- `answers_max_nst_o3m_debug.json`：样例答案的详细调试输出

## 运行系统

如需在该测试集上运行系统：

1. 在本目录解压所需压缩包：
   - `databases.zip` —— 流水线运行所需的预处理数据
   - `debug_data.zip`（可选）—— 用于调试的中间输出文件

2. 参照仓库根目录下主文件 `README.md` 中的部署与使用说明进行操作