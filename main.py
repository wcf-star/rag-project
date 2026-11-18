# 用 click 定义命令行命令，把命令行参数映射到 Pipeline 各阶段。
import click
from pathlib import Path
from src.pipeline import Pipeline, configs, RunConfig

@click.group()
def cli():
    """用于处理 PDF 报告和问题的流水线命令行界面。"""
    pass

# 处理解析后的报告（Markdown 分块 + 建向量库）
@cli.command()
def process_reports():
    """通过流水线各阶段处理已解析的报告（Markdown 分块 + 建向量库）。"""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path, run_config=RunConfig())

    click.echo("正在处理已解析的报告...")
    pipeline.process_parsed_reports()

# 问答推理流水线（LLM 相关）
@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max']), default='base', help='要使用的配置预设')
def process_questions(config):
    """使用流水线处理问题。"""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)

    click.echo(f"正在处理问题（config={config}）...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()
