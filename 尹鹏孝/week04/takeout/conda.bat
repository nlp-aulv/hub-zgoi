# 导出环境配置
conda env export --no-builds | grep -v "prefix" > environment.yml

# 从 environment.yml 生成 requirements.txt
conda env export --no-builds | grep -v "prefix" | grep -E "^(name:|channels:|- )" > environment.yml
conda list -e > requirements.txt