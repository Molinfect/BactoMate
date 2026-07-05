# Minimize the console window that launched BactoMate (used by launch_windows.bat).
$ErrorActionPreference = 'SilentlyContinue'
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class BactoMateConsole {
    [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@
$hwnd = [BactoMateConsole]::GetConsoleWindow()
if ($hwnd -ne [IntPtr]::Zero) {
    [void][BactoMateConsole]::ShowWindow($hwnd, 6)  # SW_MINIMIZE
}
