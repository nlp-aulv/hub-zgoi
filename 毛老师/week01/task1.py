import sys
import platform
import pkg_resources

def get_installed_packages():
    """获取所有已安装的包及其版本。"""
    installed_packages = {p.key: p.version for p in pkg_resources.working_set}
    return installed_packages

def print_environment_info():
    """打印Python环境和常用库的信息。"""
    print("--- 系统和Python环境 ---")
    print(f"操作系统: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python 版本: {sys.version}")
    print(f"当前工作目录: {sys.path[0]}")
    print("-" * 30)

    print("\n--- 已安装的常用库 ---")
    required_packages = {
        'jieba': 'jieba',
        'scikit-learn': 'scikit-learn',
        'pytorch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'requests': 'requests',
        'matplotlib': 'matplotlib'
    }

    installed_packages = get_installed_packages()

    for name, key in required_packages.items():
        if key in installed_packages:
            print(f"{name:<20} : {installed_packages[key]}")
        else:
            print(f"{name:<20} : 未安装")

if __name__ == "__main__":
    print_environment_info()
