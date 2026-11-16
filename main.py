# 用 click 定义 5 个命令，把命令行参数映射到 Pipeline 各阶段。
import click
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs

@click.group()
def cli():
    """用于处理 PDF 报告和问题的流水线命令行界面。"""
    pass

# 下载 Docling 模型
@cli.command()
def download_models():
    """下载所需的 docling 模型。"""
    click.echo("正在下载 docling 模型...")
    Pipeline.download_docling_models()

# PDF 解析（支持并行）
@cli.command()
@click.option('--parallel/--sequential', default=True, help='以并行或顺序模式运行解析')
@click.option('--chunk-size', default=2, help='每个 worker 处理的 PDF 数量')
@click.option('--max-workers', default=10, help='并行 worker 进程数量')
def parse_pdfs(parallel, chunk_size, max_workers):
    """解析 PDF 报告，可选择是否并行处理。"""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"正在解析 PDF 报告（parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers}）")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

# 表格序列化（支持并行）
@cli.command()
@click.option('--max-workers', default=1, help='表格序列化的 worker 数量')
def serialize_tables(max_workers):
    """使用并行线程序列化已解析报告中的表格。"""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"正在序列化表格（max_workers={max_workers}）...")
    pipeline.serialize_tables(max_workers=max_workers)
# 处理解析后的报表（预处理流水线）
@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='要使用的配置预设')
def process_reports(config):
    """通过流水线各阶段处理已解析的报告。"""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"正在处理已解析的报告（config={config}）...")
    pipeline.process_parsed_reports()

# 问答推理流水线（LLM 相关）
@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking']), default='base', help='要使用的配置预设')
def process_questions(config):
    """使用流水线处理问题。"""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"正在处理问题（config={config}）...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()