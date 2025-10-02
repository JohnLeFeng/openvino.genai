import os
import sys
from pathlib import Path

def get_human_readable_size(size_in_bytes):
    """Converts a size in bytes to a human-readable format (KB, MB, GB)."""
    if size_in_bytes is None:
        return "N/A"
    if size_in_bytes < 1024:
        return f"{size_in_bytes} Bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_in_bytes / (1024**3):.2f} GB"

def find_dll_sizes(dll_files):
    total_size = 0

    for dll_path in dll_files:
        try:
            dll_path = Path(dll_path)
            # Get the size of the file in bytes
            file_size = dll_path.stat().st_size
            total_size += file_size
            # Format the size to be more readable
            readable_size = get_human_readable_size(file_size)
            # Print the file path and its size, aligned for readability
            print(f"{str(dll_path.name):<40} {readable_size}")
        except FileNotFoundError:
            print(f"Could not analyze (file may have been moved): {dll_path.name}")
        except Exception as e:
            print(f"An error occurred with {dll_path.name}: {e}")

    print("\n" + "="*50)
    print(f"Found {len(dll_files)} DLL files.")
    print(f"Total size: {get_human_readable_size(total_size)}")
    print("="*50)


def main():
    gpu_dll_list_0 = [
        "C:\\windows\\SYSTEM32\\ntdll.dll",
        "C:\\windows\\System32\\KERNEL32.DLL",
        "C:\\windows\\System32\\KERNELBASE.dll",
        "C:\\windows\\System32\\ucrtbase.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\openvino_genai.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino.dll",
        "C:\\windows\\SYSTEM32\\MSVCP140.dll",
        "C:\\windows\\SYSTEM32\\VCRUNTIME140.dll",
        "C:\\windows\\System32\\SHLWAPI.dll",
        "C:\\windows\\SYSTEM32\\VCRUNTIME140_1.dll",
        "C:\\windows\\System32\\msvcrt.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbb12.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbbmalloc.dll",
        "C:\\windows\\System32\\umppc19811.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\openvino_tokenizers.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\icuuc70.dll",
        "C:\\windows\\System32\\ADVAPI32.dll",
        "C:\\windows\\System32\\sechost.dll",
        "C:\\windows\\System32\\bcrypt.dll",
        "C:\\windows\\System32\\RPCRT4.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\icudt70.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_ir_frontend.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_auto_batch_plugin.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_cpu_plugin.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbbbind_2_5.dll",
        "C:\\windows\\SYSTEM32\\kernel.appcore.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_gpu_plugin.dll",
        "C:\\windows\\System32\\SETUPAPI.dll",
        "C:\\windows\\SYSTEM32\\OpenCL.dll",
        "C:\\windows\\System32\\msvcp_win.dll",
        "C:\\windows\\System32\\combase.dll",
        "C:\\windows\\SYSTEM32\\cfgmgr32.dll",
        "C:\\windows\\System32\\gdi32.dll",
        "C:\\windows\\System32\\win32u.dll",
        "C:\\windows\\System32\\gdi32full.dll",
        "C:\\windows\\System32\\USER32.dll",
        "C:\\windows\\System32\\IMM32.DLL",
        "C:\\windows\\SYSTEM32\\dxcore.dll",
        "C:\\windows\\System32\\bcryptPrimitives.dll",
        "C:\\windows\\System32\\clbcatq.dll",
        "C:\\windows\\SYSTEM32\\dxgi.dll",
        "C:\\windows\\SYSTEM32\\directxdatabasehelper.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdrcl64.dll",
        "C:\\windows\\System32\\WS2_32.dll",
        "C:\\windows\\System32\\ole32.dll",
        "C:\\windows\\System32\\SHELL32.dll",
        "C:\\windows\\System32\\wintypes.dll",
        "C:\\windows\\SYSTEM32\\WINMM.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdgmm64.dll",
        "C:\\windows\\SYSTEM32\\windows.storage.dll",
        "C:\\windows\\System32\\SHCORE.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdfcl64.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igc64.dll",
        "C:\\windows\\SYSTEM32\\DEVOBJ.dll",
        "C:\\windows\\System32\\WINTRUST.dll",
        "C:\\windows\\System32\\CRYPT32.dll",
        "C:\\windows\\SYSTEM32\\MSASN1.dll",
    ]

    gpu_dll_list_1 = [
        "C:\\windows\\SYSTEM32\\ntdll.dll",
        "C:\\windows\\System32\\KERNEL32.DLL",
        "C:\\windows\\System32\\KERNELBASE.dll",
        "C:\\windows\\System32\\ucrtbase.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\openvino_genai.dll",
        "C:\\windows\\SYSTEM32\\MSVCP140.dll",
        "C:\\windows\\SYSTEM32\\VCRUNTIME140.dll",
        "C:\\windows\\SYSTEM32\\VCRUNTIME140_1.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbb12.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino.dll",
        "C:\\windows\\System32\\SHLWAPI.dll",
        "C:\\windows\\System32\\msvcrt.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbbmalloc.dll",
        "C:\\windows\\System32\\umppc19811.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\openvino_tokenizers.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\icuuc70.dll",
        "C:\\windows\\System32\\ADVAPI32.dll",
        "C:\\windows\\System32\\sechost.dll",
        "C:\\windows\\System32\\bcrypt.dll",
        "C:\\windows\\System32\\RPCRT4.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\icudt70.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_ir_frontend.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_auto_batch_plugin.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_cpu_plugin.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbbbind_2_5.dll",
        "C:\\windows\\SYSTEM32\\kernel.appcore.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_gpu_plugin.dll",
        "C:\\windows\\System32\\SETUPAPI.dll",
        "C:\\windows\\SYSTEM32\\OpenCL.dll",
        "C:\\windows\\System32\\msvcp_win.dll",
        "C:\\windows\\System32\\combase.dll",
        "C:\\windows\\SYSTEM32\\cfgmgr32.dll",
        "C:\\windows\\System32\\gdi32.dll",
        "C:\\windows\\System32\\win32u.dll",
        "C:\\windows\\System32\\gdi32full.dll",
        "C:\\windows\\System32\\USER32.dll",
        "C:\\windows\\System32\\IMM32.DLL",
        "C:\\windows\\SYSTEM32\\dxcore.dll",
        "C:\\windows\\System32\\bcryptPrimitives.dll",
        "C:\\windows\\System32\\clbcatq.dll",
        "C:\\windows\\SYSTEM32\\dxgi.dll",
        "C:\\windows\\SYSTEM32\\directxdatabasehelper.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdrcl64.dll",
        "C:\\windows\\System32\\WS2_32.dll",
        "C:\\windows\\System32\\ole32.dll",
        "C:\\windows\\System32\\SHELL32.dll",
        "C:\\windows\\System32\\wintypes.dll",
        "C:\\windows\\SYSTEM32\\WINMM.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdgmm64.dll",
        "C:\\windows\\SYSTEM32\\windows.storage.dll",
        "C:\\windows\\System32\\SHCORE.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdfcl64.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igc64.dll",
        "C:\\windows\\SYSTEM32\\DEVOBJ.dll",
        "C:\\windows\\System32\\WINTRUST.dll",
        "C:\\windows\\System32\\CRYPT32.dll",
        "C:\\windows\\SYSTEM32\\MSASN1.dll",
    ]

    gpu_dll_list_2 = [
        "C:\\windows\\SYSTEM32\\ntdll.dll",
        "C:\\windows\\System32\\KERNEL32.DLL",
        "C:\\windows\\System32\\KERNELBASE.dll",
        "C:\\windows\\System32\\ucrtbase.dll",
        "C:\\windows\\SYSTEM32\\MSVCP140.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\openvino_genai.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino.dll",
        "C:\\windows\\SYSTEM32\\VCRUNTIME140_1.dll",
        "C:\\windows\\SYSTEM32\\VCRUNTIME140.dll",
        "C:\\windows\\System32\\SHLWAPI.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbb12.dll",
        "C:\\windows\\System32\\msvcrt.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbbmalloc.dll",
        "C:\\windows\\System32\\umppc19811.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\openvino_tokenizers.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\icuuc70.dll",
        "C:\\windows\\System32\\ADVAPI32.dll",
        "C:\\windows\\System32\\sechost.dll",
        "C:\\windows\\System32\\bcrypt.dll",
        "C:\\windows\\System32\\RPCRT4.dll",
        "C:\\Users\\johnfeng\\Documents\\forks\\openvino.genai\\build-x86_64_t2i_memory\\samples\\cpp\\image_generation\\Release\\icudt70.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_ir_frontend.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_auto_batch_plugin.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_cpu_plugin.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\3rdparty\\tbb\\bin\\tbbbind_2_5.dll",
        "C:\\windows\\SYSTEM32\\kernel.appcore.dll",
        "C:\\Users\\johnfeng\\Downloads\\openvino_toolkit_windows_2025.4.0.dev20250929_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_gpu_plugin.dll",
        "C:\\windows\\System32\\SETUPAPI.dll",
        "C:\\windows\\SYSTEM32\\OpenCL.dll",
        "C:\\windows\\System32\\msvcp_win.dll",
        "C:\\windows\\System32\\combase.dll",
        "C:\\windows\\SYSTEM32\\cfgmgr32.dll",
        "C:\\windows\\System32\\gdi32.dll",
        "C:\\windows\\System32\\win32u.dll",
        "C:\\windows\\System32\\gdi32full.dll",
        "C:\\windows\\System32\\USER32.dll",
        "C:\\windows\\System32\\IMM32.DLL",
        "C:\\windows\\SYSTEM32\\dxcore.dll",
        "C:\\windows\\System32\\bcryptPrimitives.dll",
        "C:\\windows\\System32\\clbcatq.dll",
        "C:\\windows\\SYSTEM32\\dxgi.dll",
        "C:\\windows\\SYSTEM32\\directxdatabasehelper.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdrcl64.dll",
        "C:\\windows\\System32\\WS2_32.dll",
        "C:\\windows\\System32\\ole32.dll",
        "C:\\windows\\System32\\SHELL32.dll",
        "C:\\windows\\System32\\wintypes.dll",
        "C:\\windows\\SYSTEM32\\WINMM.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdgmm64.dll",
        "C:\\windows\\SYSTEM32\\windows.storage.dll",
        "C:\\windows\\System32\\SHCORE.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igdfcl64.dll",
        "C:\\windows\\System32\\DriverStore\\FileRepository\\iigd_dch.inf_amd64_0fa5351289a5f61a\\igc64.dll",
        "C:\\windows\\SYSTEM32\\DEVOBJ.dll",
        "C:\\windows\\System32\\WINTRUST.dll",
        "C:\\windows\\System32\\CRYPT32.dll",
        "C:\\windows\\SYSTEM32\\MSASN1.dll",
    ]

    print("\nGPU DLL list 0\n")
    find_dll_sizes(gpu_dll_list_0)

    print("\nGPU DLL list 1\n")
    find_dll_sizes(gpu_dll_list_1)

    print("\nGPU DLL list 2\n")
    find_dll_sizes(gpu_dll_list_2)

if __name__ == "__main__":
    sys.exit(main())