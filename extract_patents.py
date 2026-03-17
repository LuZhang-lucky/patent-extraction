import openai
import os
import json
from pathlib import Path
import time

print("脚本开始运行了！") # 👈 新加这一行

# 1. 设置你的API密钥
api_key = os.getenv("OPENAI_API_KEY")

# 2. 设置PDF文件夹路径（改成你的实际路径）
pdf_folder = Path("/Users/lulu/Desktop/patent_test/raw_pdfs")  # Mac路径示例

# 3. 获取所有PDF文件
pdf_files = list(pdf_folder.glob("*.pdf"))
print(f"找到 {len(pdf_files)} 个PDF文件")

# 4. 提取提示词（根据我们之前优化的版本）
extract_prompt = """
Extract the following metadata from this Swedish patent document.
Keep all extracted text in original Swedish language.
Output field names in Swedish as specified below.

Output in JSON format:
{
  "Patentnummer": "string (patent number)",
  "Titel": "string (invention title in Swedish)",
  "Sökande": "string (applicant name)",
  "Uppfinnare": [
    {
      "namn": "string (inventor name)",
      "ort": "string (city if available, otherwise null)",
      "land": "string (country if available, otherwise null)"
    }
  ],
  "Ansökningsdatum": "YYYY-MM-DD",
  "Publiceringsdatum": "YYYY-MM-DD",
  "IPC-klass": "string or null"
}

Rules:
- Uppfinnare should be an array, even if only one inventor
- If inventor location is not specified, use null
- If a field is not found, use null
- Date format: YYYY-MM-DD
"""

# 5. 批量处理PDF
results = []
for i, pdf_path in enumerate(pdf_files[:5]):  # 先处理前5份
    print(f"处理第 {i+1} 份: {pdf_path.name}")
    
    try:
        # 上传文件
        with open(pdf_path, "rb") as f:
            file = client.files.create(
                file=f,
                purpose="assistants"  # 用于文件处理的purpose
            )
        
        # 创建消息并调用模型
        response = client.chat.completions.create(
            model="gpt-4o",  # 用你经费支持的旗舰模型
            messages=[
                {"role": "system", "content": "你是一个瑞典专利信息提取专家。"},
                {"role": "user", "content": [
                    {"type": "text", "text": extract_prompt},
                    {"type": "file", "file_id": file.id}
                ]}
            ],
            response_format={"type": "json_object"}  # 强制输出JSON
        )
        
        # 解析结果
        result = json.loads(response.choices[0].message.content)
        result["_filename"] = pdf_path.name  # 加一列文件名
        
        results.append(result)
        print(f"  ✅ 成功提取")
        
        # 删除服务端的文件（节省空间）
        client.files.delete(file.id)
        
        # 稍微停顿一下，避免API限流
        time.sleep(1)
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        results.append({
            "_filename": pdf_path.name,
            "_error": str(e)
        })

# 6. 保存结果
output_file = pdf_folder / "extraction_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 全部完成！结果保存在: {output_file}")

# 7. 可选：转成Excel（需要安装pandas）
try:
    import pandas as pd
    df = pd.DataFrame(results)
    excel_file = pdf_folder / "extraction_results.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✅ Excel文件已生成: {excel_file}")
except ImportError:
    print("提示: 如需生成Excel，请安装pandas: pip install pandas openpyxl")