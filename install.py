#!/usr/bin/env python3
"""
Script instalasi DeepGuard yang sudah diperbaiki untuk Windows.
Mengganti facenet-pytorch dengan mtcnn yang lebih ringan.
"""
import subprocess, sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def pip(packages, extra_args=None):
    cmd = [sys.executable, '-m', 'pip', 'install'] + packages + ['-q']
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr[:200]

print('🔧 Instalasi DeepGuard (Windows-compatible)\n')

steps = [
    # (nama, packages, extra_args)
    ('PyTorch (CPU)',     ['torch', 'torchvision'],
     ['--index-url', 'https://download.pytorch.org/whl/cpu']),

    ('timm',             ['timm==0.9.12'],             None),
    ('albumentations',   ['albumentations>=1.4.0'],    None),
    ('OpenCV',           ['opencv-python'],             None),
    ('Pillow + numpy',   ['Pillow', 'numpy'],           None),
    ('scipy + sklearn',  ['scipy', 'scikit-learn'],     None),
    ('scikit-image',     ['scikit-image'],              None),
    ('matplotlib',       ['matplotlib', 'seaborn'],     None),
    ('tqdm',             ['tqdm'],                      None),
    ('reportlab (PDF)',  ['reportlab'],                 None),
    ('mediapipe',        ['mediapipe'],                 None),

    # Coba facenet-pytorch dulu, kalau gagal pakai mtcnn
    ('MTCNN face detector', ['mtcnn'],                 None),
]

all_ok = True
for name, pkgs, extra in steps:
    print(f'  📦 {name}...', end='', flush=True)
    ok, err = pip(pkgs, extra)
    if ok:
        print(' ✅')
    else:
        err_msg = f" (Error: {err.strip()})" if err.strip() else ""
        print(f' ⚠️  Gagal/Warning{err_msg}')
        if 'mtcnn' not in name.lower():
            all_ok = False

# Coba facenet-pytorch terpisah (opsional)
print(f'  📦 facenet-pytorch (opsional)...', end='', flush=True)
ok, err = pip(['facenet-pytorch'])
if ok:
    print(' ✅')
else:
    print(' ⏭️  Skip (pakai mtcnn sebagai gantinya)')

print()
if all_ok:
    print('✅ Instalasi selesai!')
else:
    print('⚠️  Beberapa package gagal. Jika ada kendala, coba jalankan: pip install --upgrade pip')
    print('   atau install package secara manual.')

print('\n📋 Verifikasi instalasi...')
packages_to_check = [
    ('torch',        'PyTorch'),
    ('timm',         'timm'),
    ('cv2',          'OpenCV'),
    ('albumentations','albumentations'),
    ('PIL',          'Pillow'),
    ('numpy',        'numpy'),
    ('scipy',        'scipy'),
    ('sklearn',      'scikit-learn'),
    ('reportlab',    'reportlab'),
    ('mediapipe',    'mediapipe'),
    ('mtcnn',        'mtcnn'),
]

ok_count = 0
for import_name, display_name in packages_to_check:
    try:
        __import__(import_name)
        print(f'  ✅ {display_name}')
        ok_count += 1
    except ImportError:
        print(f'  ❌ {display_name} (tidak tersedia)')

print(f'\n  {ok_count}/{len(packages_to_check)} package tersedia')

if ok_count >= 7:
    print('\n🚀 Siap dijalankan!')
    print('   python deepguard.py --input foto.jpg --report')
else:
    print('\n⚠️  Terlalu banyak package yang gagal')
    print('   Coba jalankan: pip install torch timm opencv-python reportlab numpy')
