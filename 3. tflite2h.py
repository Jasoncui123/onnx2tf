

# tflite2h.py
# 把 tflite 转成 .h 文件
input_path = "saved_model_p/papagei_p_dynamic_range_quant.tflite"
output_path = "saved_model_p/model_data.h"

with open(input_path, "rb") as f:
    data = f.read()

arr_name = "model_data"
arr_len = len(data)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
    f.write(f"const unsigned char {arr_name}[] = {{\n")
    
    s = ""
    for i, b in enumerate(data):
        s += f"0x{b:02x}, "
        if (i+1) % 16 == 0:
            f.write("    " + s + "\n")
            s = ""
    if s:
        f.write("    " + s + "\n")
    
    f.write(f"}};\n\n")
    f.write(f"const int {arr_name}_len = {arr_len};\n\n")
    f.write("#endif\n")

print(f"✅ 转换完成！输出：{output_path}")
