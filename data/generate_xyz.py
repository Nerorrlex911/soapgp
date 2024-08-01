from openbabel import pybel
import sys

def sdf_to_xyz(input_file, output_file):
    # 使用 pybel 加载 SDF 文件
    mols = pybel.readfile("sdf", input_file)

    # 将每个分子转换为 XYZ 格式并写入文件
    with open(output_file, "w") as xyz_file:
        for mol in mols:
            # 写入分子的 XYZ 格式
            xyz_file.write(mol.write("xyz"))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_xyz.py input.sdf output.xyz")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    sdf_to_xyz(input_file, output_file)
    print(f"Conversion complete: {input_file} to {output_file}")
