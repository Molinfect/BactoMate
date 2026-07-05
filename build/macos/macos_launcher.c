/*
 * Native BactoMate.app launcher for macOS.
 *
 * Runs the bundled GUI inside this Mach-O process via Python.framework
 * instead of exec'ing a separate python3 binary (which makes macOS show Python
 * as a second Dock app).
 */
#include <mach-o/dyld.h>

#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef BACTOMATE_LINK_PYTHON
#include <Python.h>
#endif

static void append_log(const char *resources_dir, const char *fmt, ...) {
    char log_dir[PATH_MAX];
    char log_path[PATH_MAX];
    snprintf(log_dir, sizeof(log_dir), "%s/Logs", resources_dir);
    mkdir(log_dir, 0755);
    snprintf(log_path, sizeof(log_path), "%s/Logs/launch.log", resources_dir);

    FILE *fp = fopen(log_path, "a");
    if (!fp) {
        return;
    }

    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char stamp[32];
    strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", &tm_now);
    fprintf(fp, "%s: ", stamp);

    va_list args;
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
    fputc('\n', fp);
    fclose(fp);
}

static bool resolve_paths(
    char *exe_path,
    size_t exe_path_len,
    char *resources_dir,
    size_t resources_dir_len,
    char *main_py,
    size_t main_py_len,
    char *python_bin,
    size_t python_bin_len,
    char *qt_plugins,
    size_t qt_plugins_len
) {
    uint32_t size = (uint32_t)exe_path_len;
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        return false;
    }

    char resolved[PATH_MAX];
    if (!realpath(exe_path, resolved)) {
        strncpy(resolved, exe_path, sizeof(resolved));
        resolved[sizeof(resolved) - 1] = '\0';
    }
    strncpy(exe_path, resolved, exe_path_len);
    exe_path[exe_path_len - 1] = '\0';

    const char *marker = "/Contents/MacOS/";
    char *found = strstr(exe_path, marker);
    if (!found) {
        return false;
    }

    size_t prefix_len = (size_t)(found - exe_path);
    if (snprintf(resources_dir, resources_dir_len, "%.*s/Contents/Resources", (int)prefix_len, exe_path)
        >= (int)resources_dir_len) {
        return false;
    }

    snprintf(main_py, main_py_len, "%s/main.py", resources_dir);
    snprintf(python_bin, python_bin_len, "%s/venv/bin/python3", resources_dir);
    snprintf(
        qt_plugins,
        qt_plugins_len,
        "%s/venv/lib/python%s/site-packages/PyQt6/Qt6/plugins",
        resources_dir,
        BACTOMATE_PY_MM
    );
    return true;
}

static bool bundled_framework_paths(
    const char *exe_path,
    char *framework_root,
    size_t framework_root_len,
    char *python_home,
    size_t python_home_len,
    char *python_executable,
    size_t python_executable_len
) {
    const char *marker = "/Contents/MacOS/";
    char *found = strstr((char *)exe_path, marker);
    if (!found) {
        return false;
    }

    size_t prefix_len = (size_t)(found - exe_path);
    if (snprintf(
            framework_root,
            framework_root_len,
            "%.*s/Contents/Frameworks/Python.framework",
            (int)prefix_len,
            exe_path
        ) >= (int)framework_root_len) {
        return false;
    }
    if (snprintf(
            python_home,
            python_home_len,
            "%s/Versions/%s",
            framework_root,
            BACTOMATE_PY_MM
        ) >= (int)python_home_len) {
        return false;
    }
    if (snprintf(
            python_executable,
            python_executable_len,
            "%s/bin/python%s",
            python_home,
            BACTOMATE_PY_MM
        ) >= (int)python_executable_len) {
        return false;
    }

    struct stat st;
    if (stat(python_home, &st) != 0 || !S_ISDIR(st.st_mode)) {
        return false;
    }
    return true;
}

static void configure_runtime_env(
    const char *exe_path,
    const char *resources_dir,
    const char *qt_plugins
) {
    char venv_bin[PATH_MAX];
    char venv_site[PATH_MAX];
    char qt_frameworks[PATH_MAX];
    char framework_root[PATH_MAX];
    char python_home[PATH_MAX];
    char python_executable[PATH_MAX];
    snprintf(venv_bin, sizeof(venv_bin), "%s/venv/bin", resources_dir);
    snprintf(venv_site, sizeof(venv_site), "%s/venv/lib/python%s/site-packages", resources_dir, BACTOMATE_PY_MM);
    snprintf(qt_frameworks, sizeof(qt_frameworks), "%s/venv/lib/python%s/site-packages/PyQt6/Qt6/lib", resources_dir, BACTOMATE_PY_MM);

    unsetenv("PYTHONHOME");
    unsetenv("QT_QPA_PLATFORM_PLUGIN_PATH");

    bool has_framework = bundled_framework_paths(
        exe_path,
        framework_root,
        sizeof(framework_root),
        python_home,
        sizeof(python_home),
        python_executable,
        sizeof(python_executable)
    );
    char dyld_framework_path[PATH_MAX * 3];
    dyld_framework_path[0] = '\0';
    if (has_framework) {
        // Bundled venvs depend on the copied Python.framework for stdlib modules.
        setenv("PYTHONHOME", python_home, 1);
        snprintf(dyld_framework_path, sizeof(dyld_framework_path), "%s:%s", framework_root, qt_frameworks);
    } else {
        snprintf(dyld_framework_path, sizeof(dyld_framework_path), "%s", qt_frameworks);
    }
    const char *existing_dyld = getenv("DYLD_FRAMEWORK_PATH");
    if (existing_dyld && existing_dyld[0]) {
        char dyld_path[PATH_MAX * 4];
        snprintf(dyld_path, sizeof(dyld_path), "%s:%s", dyld_framework_path, existing_dyld);
        setenv("DYLD_FRAMEWORK_PATH", dyld_path, 1);
    } else {
        setenv("DYLD_FRAMEWORK_PATH", dyld_framework_path, 1);
    }

    setenv("PYTHONPATH", venv_site, 1);
    struct stat py_exec_st;
    if (has_framework && stat(python_executable, &py_exec_st) == 0) {
        setenv("PYTHONEXECUTABLE", python_executable, 1);
    } else {
        setenv("PYTHONEXECUTABLE", exe_path, 1);
    }
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("QT_PLUGIN_PATH", qt_plugins, 1);
    setenv("QT_FRAMEWORK_PATH", qt_frameworks, 1);
    setenv("QT_MAC_WANTS_LAYER", "1", 1);
    setenv("QT_AUTO_SCREEN_SCALE_FACTOR", "1", 1);

    char path_env[PATH_MAX * 2];
    const char *existing = getenv("PATH");
    if (existing && existing[0]) {
        snprintf(path_env, sizeof(path_env), "%s:/usr/bin:/bin:/usr/sbin:/sbin:%s", venv_bin, existing);
    } else {
        snprintf(path_env, sizeof(path_env), "%s:/usr/bin:/bin:/usr/sbin:/sbin", venv_bin);
    }
    setenv("PATH", path_env, 1);
}

#ifdef BACTOMATE_LINK_PYTHON
static int run_inprocess_python(const char *exe_path, const char *main_py, int argc, char *argv[]) {
    wchar_t *program = Py_DecodeLocale(exe_path, NULL);
    if (!program) {
        return 1;
    }
    Py_SetProgramName(program);

    int py_argc = argc > 1 ? argc + 1 : 2;
    wchar_t **py_argv = calloc((size_t)py_argc + 1, sizeof(wchar_t *));
    if (!py_argv) {
        PyMem_RawFree(program);
        return 1;
    }

    py_argv[0] = Py_DecodeLocale(exe_path, NULL);
    py_argv[1] = Py_DecodeLocale(main_py, NULL);
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            py_argv[i + 1] = Py_DecodeLocale(argv[i], NULL);
        }
    }

    for (int i = 0; i < py_argc; i++) {
        if (!py_argv[i]) {
            for (int j = 0; j < py_argc; j++) {
                if (py_argv[j]) {
                    PyMem_RawFree(py_argv[j]);
                }
            }
            free(py_argv);
            PyMem_RawFree(program);
            return 1;
        }
    }

    int rc = Py_Main(py_argc, py_argv);
    for (int i = 0; i < py_argc; i++) {
        PyMem_RawFree(py_argv[i]);
    }
    free(py_argv);
    PyMem_RawFree(program);
    return rc;
}
#endif

static int run_exec_fallback(const char *exe_path, const char *python_bin, const char *main_py) {
    char *const argv[] = { (char *)exe_path, (char *)main_py, NULL };
    execv(python_bin, argv);
    return 1;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    char exe_path[PATH_MAX];
    char resources_dir[PATH_MAX];
    char main_py[PATH_MAX];
    char python_bin[PATH_MAX];
    char qt_plugins[PATH_MAX];

    if (!resolve_paths(exe_path, sizeof(exe_path), resources_dir, sizeof(resources_dir), main_py, sizeof(main_py), python_bin, sizeof(python_bin), qt_plugins, sizeof(qt_plugins))) {
        fprintf(stderr, "BactoMate launcher: could not resolve app bundle paths.\n");
        return 1;
    }

    struct stat st;
    if (stat(main_py, &st) != 0) {
        fprintf(stderr, "BactoMate launcher: missing main.py in bundle.\n");
        return 1;
    }
    if (stat(python_bin, &st) != 0) {
        fprintf(stderr, "BactoMate launcher: missing bundled python3.\n");
        return 1;
    }
    if (stat(qt_plugins, &st) != 0) {
        fprintf(stderr, "BactoMate launcher: missing Qt plugins.\n");
        return 1;
    }

    configure_runtime_env(exe_path, resources_dir, qt_plugins);
    append_log(
        resources_dir,
        "Native launcher starting in-process Python: %s (PYTHONHOME=%s)",
        main_py,
        getenv("PYTHONHOME") ? getenv("PYTHONHOME") : "<unset>"
    );

#ifdef BACTOMATE_LINK_PYTHON
    return run_inprocess_python(exe_path, main_py, argc, argv);
#else
    append_log(resources_dir, "Native launcher fallback: exec bundled python3");
    return run_exec_fallback(exe_path, python_bin, main_py);
#endif
}
