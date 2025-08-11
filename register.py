import os
import json
import argparse
import logging
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import sys

def stream_detokenize_climblab(output_file, max_tokens=1000000):
    """
    使用流式读取方式处理ClimbLab数据集：
    - 流式读取nvidia/ClimbLab数据集
    - 使用GPT2 tokenizer解码tokens
    - 构造JSON格式输出
    - 确保总token数量不超过max_tokens（默认1M）
    """
    logging.info("开始流式读取ClimbLab数据集...")
    
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 流式加载数据集
    ds = load_dataset("nvidia/ClimbLab", streaming=True)
    
    # 初始化计数器和结果列表
    total_tokens = 0
    processed_records = []
    record_count = 0
    
    # 获取训练集的迭代器
    train_iter = iter(ds['train'])
    
    logging.info(f"开始处理数据，目标最大token数: {max_tokens:,}")
    
    try:
        # 创建进度条
        pbar = tqdm(desc="处理记录", unit="records")
        
        while total_tokens < max_tokens:
            try:
                # 获取下一条记录
                record = next(train_iter)
                
                # 提取tokens和token_count
                tokens = record.get("tokens", [])
                token_count = record.get("token_count", len(tokens))
                
                # 检查是否会超过限制
                if total_tokens + token_count > max_tokens:
                    logging.info(f"添加当前记录会超过token限制，停止处理")
                    break
                
                # 解码tokens为文本
                try:
                    text = tokenizer.decode(tokens)
                except Exception as e:
                    logging.warning(f"解码第{record_count+1}条记录的tokens时出错: {e}")
                    continue
                
                # 构造记录
                processed_record = {
                    "token_count": token_count,
                    "text": text
                }
                
                processed_records.append(processed_record)
                total_tokens += token_count
                record_count += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'tokens': f'{total_tokens:,}',
                    'records': record_count
                })
                pbar.update(1)
                
                # 每处理1000条记录打印一次状态
                if record_count % 1000 == 0:
                    logging.info(f"已处理 {record_count:,} 条记录，累计tokens: {total_tokens:,}")
                
            except StopIteration:
                logging.info("数据集已读取完毕")
                break
            except Exception as e:
                logging.error(f"处理记录时出错: {e}")
                continue
        
        pbar.close()
        
    except KeyboardInterrupt:
        logging.info("用户中断处理")
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        return False
    
    # 构造符合LMFlow要求的JSON输出格式
    output_data = {
        "type": "text_only",
        "instances": [{"text": record["text"]} for record in processed_records]
    }
    
    # 写入JSON文件
    logging.info(f"写入结果到 {output_file}...")
    try:
        # 如果输出文件所在目录不存在，则创建
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"处理完成！")
        logging.info(f"- 处理记录数: {len(processed_records):,}")
        logging.info(f"- 总token数: {total_tokens:,}")
        logging.info(f"- 输出文件: {output_file}")
        logging.info(f"- 文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        # 将统计信息保存到_meta.json文件
        meta_info = {
            "processed_records": len(processed_records),
            "total_tokens": total_tokens,
            "output_file": output_file,
            "output_file_size_MB": round(os.path.getsize(output_file) / 1024 / 1024, 2)
        }
        meta_file = os.path.splitext(output_file)[0] + "_meta.json"
        try:
            with open(meta_file, 'w', encoding='utf-8') as mf:
                json.dump(meta_info, mf, ensure_ascii=False, indent=2)
            logging.info(f"统计信息已保存到 {meta_file}")
        except Exception as e:
            logging.error(f"写入_meta.json时出错: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"写入文件时出错: {e}")
        return False

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    parser = argparse.ArgumentParser(
        description="流式处理ClimbLab数据集，解码tokens并构造JSON输出，限制总token数量"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="climblab_detokenized.json",
        help="输出JSON文件路径（默认: climblab_detokenized.json）"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=1000000,
        help="最大token数量限制（默认: 1,000,000）"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 执行处理
    success = stream_detokenize_climblab(args.output_file, args.max_tokens)
    
    if success:
        print(f"\n✅ 处理成功完成！输出文件: {args.output_file}")
        sys.exit(0)
    else:
        print(f"\n❌ 处理失败，请检查日志信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
