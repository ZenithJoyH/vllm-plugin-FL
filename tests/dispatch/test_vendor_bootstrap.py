"""Focused bootstrap tests; stdlib unit tests also run without Torch or a PPU."""

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'ops': _ROOT / 'vllm_fl/ops/_C_ops_registry.py',
    'root': _ROOT / 'vllm_fl/__init__.py',
    'vendor': _ROOT / 'vllm_fl/dispatch/backends/vendor/__init__.py',
    'thead': _ROOT / 'vllm_fl/dispatch/backends/vendor/thead/__init__.py',
    'bootstrap': _ROOT / 'vllm_fl/dispatch/backends/vendor/thead/bootstrap.py',
}


def load_source(key, name):
    options = {'submodule_search_locations': []} if key in ('vendor', 'thead') else {}
    spec = importlib.util.spec_from_file_location(name, SOURCES[key], **options)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BootstrapTests(unittest.TestCase):
    def test_public_entrypoint_delegates_native_setup_to_custom_ops(self):
        source = SOURCES['root'].read_text()
        self.assertNotIn('_load_thead_native_extensions', source)
        self.assertNotIn('PPU_SDK', source)
        self.assertNotIn('vendor.thead', source)
        self.assertNotIn('initialize_native_extensions', source)
        self.assertNotIn('ops.bootstrap', source)
        register = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'register')
        calls = [n.value.func.id for n in register.body
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name)]
        self.assertEqual(calls[0], '_patch_custom_ops')
        self.assertNotIn('initialize_native_extensions', ast.unparse(register))

    def test_registry_preload_before_early_return(self):
        source = SOURCES['ops'].read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'register_op_schemas')
        vendor = types.ModuleType('vllm_fl.dispatch.backends.vendor.thead.bootstrap')
        vendor.initialize_native_extensions = Mock()
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<registry>', 'exec'), ns)
        ns['register_op_schemas']._lib = object()
        with patch.dict(sys.modules, {vendor.__name__: vendor}):
            ns['register_op_schemas']()
            vendor.initialize_native_extensions.assert_called_once_with()
            vendor.initialize_native_extensions.side_effect = RuntimeError('native failed')
            with self.assertRaisesRegex(RuntimeError, 'native failed'):
                ns['register_op_schemas']()

    def test_vendor_package_has_no_extra_registry(self):
        source = SOURCES['vendor'].read_text()
        self.assertNotIn('_NATIVE_EXTENSION_INITIALIZERS', source)
        self.assertNotIn('def initialize_native_extensions', source)

    def test_non_ppu_does_not_import_native_loader(self):
        module = load_source('bootstrap', 'isolated_thead.bootstrap')
        with patch.dict(os.environ, {}, clear=True):
            with patch('builtins.__import__', side_effect=AssertionError('unexpected import')):
                module.initialize_native_extensions()

    def test_ppu_delegates_and_preserves_native_failure(self):
        module = load_source('bootstrap', 'isolated_thead.bootstrap')
        native = types.ModuleType('isolated_thead.impl.native_extensions')
        native.load_all_native_extensions = Mock()
        with patch.dict(sys.modules, {native.__name__: native}):
            with patch.dict(os.environ, {'PPU_SDK': '/test/sdk'}):
                module.initialize_native_extensions()
                native.load_all_native_extensions.assert_called_once_with()
                native.load_all_native_extensions.side_effect = RuntimeError('missing SO')
                with self.assertRaisesRegex(RuntimeError, 'missing SO'):
                    module.initialize_native_extensions()

    def test_thead_exports_remain_lazy_and_compatible(self):
        module = load_source('thead', 'isolated_thead')
        self.assertNotIn('TheadBackend', module.__dict__)
        backend = types.ModuleType('isolated_thead.thead')
        backend.TheadBackend = type('TheadBackend', (), {})
        with patch.dict(sys.modules, {backend.__name__: backend}):
            self.assertIs(module.TheadBackend, backend.TheadBackend)
            self.assertIs(module.TheadBackend, backend.TheadBackend)
        with self.assertRaises(AttributeError):
            module.unknown_export


def runtime_checks(root):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
    env.pop('PPU_SDK', None)
    subprocess.run([sys.executable, '-B', '-c', '''
import sys, torch
import vllm_fl
from vllm_fl.dispatch.backends.vendor.thead.bootstrap import initialize_native_extensions
before = set(torch.ops.loaded_libraries)
initialize_native_extensions()
initialize_native_extensions()
assert set(torch.ops.loaded_libraries) == before
assert 'vllm_fl.dispatch.backends.vendor.thead.impl.native_extensions' not in sys.modules
assert 'vllm_fl.dispatch.backends.vendor.thead.thead' not in sys.modules
print('NON_PPU_IMPORT_ISOLATION_OK')
'''], env=env, check=True, timeout=120)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
    assert 'PPU_SDK' in env, 'PPU runtime verification requires the real SDK environment'
    subprocess.run([sys.executable, '-B', '-c', '''
import sys, torch
import vllm_fl
import ast, pathlib, subprocess
path = pathlib.Path(vllm_fl.__file__)
old = subprocess.check_output(['git', '-C', str(path.parent.parent), 'show', 'HEAD:vllm_fl/__init__.py'], text=True)
def function(source):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_patch_custom_ops'))
assert function(old) == function(path.read_text())
vllm_fl._patch_custom_ops()
from vllm_fl.ops._C_ops_registry import register_op_schemas
assert getattr(register_op_schemas, '_lib', None) is None
assert 'vllm_fl.ops.bootstrap' not in sys.modules
assert vllm_fl.register() == 'vllm_fl.platform.PlatformFL'
from vllm_fl.dispatch.backends.vendor.thead.bootstrap import initialize_native_extensions
from vllm_fl.dispatch.backends.vendor.thead.impl import native_extensions
assert native_extensions._LOADED == {'cache', 'core', 'moe'}
before = set(torch.ops.loaded_libraries)
initialize_native_extensions()
initialize_native_extensions()
vllm_fl._patch_custom_ops()
assert set(torch.ops.loaded_libraries) == before
from vllm_fl.ops._C_ops_registry import register_op_schemas
assert getattr(register_op_schemas, '_lib', None) is None
assert 'vllm_fl.ops.bootstrap' not in sys.modules
assert torch._C._dispatch_has_kernel_for_dispatch_key('_C_cache_ops::concat_and_cache_mla', 'CUDA')
assert torch._C._dispatch_has_kernel_for_dispatch_key('_moe_C::moe_sum', 'CUDA')
print('PPU_EARLY_SCHEMA_AND_IDEMPOTENCY_OK')
'''], env=env, check=True, timeout=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--snapshots', type=Path)
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    if args.snapshots:
        SOURCES.update({key: args.snapshots / filename for key, filename in {
            'ops': 'ppu07-plugin-_C_ops_registry.py',
            'root': 'ppu07-plugin-package-init.py',
            'vendor': 'ppu07-plugin-vendor-init.py',
            'thead': 'ppu07-plugin-thead-init.py',
            'bootstrap': 'ppu07-plugin-thead-bootstrap.py',
        }.items()})
    else:
        root = args.root or Path(__file__).resolve().parents[2]
        SOURCES.update({
            'ops': root / 'vllm_fl/ops/_C_ops_registry.py',
            'root': root / 'vllm_fl/__init__.py',
            'vendor': root / 'vllm_fl/dispatch/backends/vendor/__init__.py',
            'thead': root / 'vllm_fl/dispatch/backends/vendor/thead/__init__.py',
            'bootstrap': root / 'vllm_fl/dispatch/backends/vendor/thead/bootstrap.py',
        })
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.runtime:
        runtime_checks(args.root)
