#!/usr/bin/env bash
#
# run_video_editor.sh
# سكربت ذكي: يحقق تثبيت المستودعات والحزم المطلوبة عند أول تشغيل،
# ثم يشغّل كود Python الرسومي (video_editor_gui.py) بنقرة واحدة دون طرفية.
#
# كيفية الاستخدام:
# - اجعل الملف قابلًا للتنفيذ: chmod +x run_video_editor.sh
# - إذا أردت، ضع مسار السكربت في مجلد ثابت، ثم أنشئ .desktop للإطلاق بالنقرة.
#

# تحديد مسار السكربت بشكل مطلق
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# دالة طباعة معلومات ملونة
echo_info() {
    echo -e "\\033[1;34m[INFO]\\033[0m $1"
}
echo_error() {
    echo -e "\\033[1;31m[ERROR]\\033[0m $1"
}

# دالة تثبيت apt إذا لزم
apt_install_if_missing() {
    PKG="$1"
    # إذا لم يكن مثبتًا
    if ! dpkg-query -W -f='${Status}' "$PKG" 2>/dev/null | grep -q "install ok installed"; then
        echo_info "تثبيت الحزمة: $PKG"
        apt-get install -y "$PKG"
        if [ $? -ne 0 ]; then
            echo_error "فشل تثبيت $PKG"
            exit 1
        fi
    else
        echo_info "الحزمة مثبتة مسبقًا: $PKG"
    fi
}

# دالة للتحقق من تفعيل add-apt-repository
ensure_add_apt_repository_available() {
    if ! command -v add-apt-repository >/dev/null 2>&1; then
        echo_info "تثبيت software-properties-common لتمكين add-apt-repository"
        apt-get install -y software-properties-common
        if [ $? -ne 0 ]; then
            echo_error "فشل تثبيت software-properties-common"
            exit 1
        fi
    else
        echo_info "add-apt-repository متوفرة"
    fi
}

# إذا تشغيل عادي (غير root) ولم يكن الوسيط --install، نبدأ مرحلة التثبيت تحت pkexec
if [ "$EUID" -ne 0 ] && [ "$1" != "--install" ]; then
    # نعيد استدعاء السكربت تحت pkexec ليرتقي إلى root ويقوم بالتثبيت.
    # ننقل DISPLAY و XAUTHORITY لضمان ظهور أي نوافذ إذا لزم (ثمة بعض التوزيعات قد تحتاج ضبط إضافي للبيئة، لكن غالبًا يعمل مباشرة).
    echo_info "سيطلب صلاحيات المسؤول لتثبيت الحزم المطلوبة..."
    # ملاحظة: pkexec قد يتجاهل بعض المتغيرات البيئية حسب Polkit policy. إذا واجهت مشاكل بالعرض، قد تحتاج إعداد PolicyKit agent أو ضبط env بدقة.
    pkexec env DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" "$SCRIPT_PATH" --install
    # بعد انتهاء التثبيت (عند عودة pkexec دون خطأ)، سنكمل التشغيل كـ user عادي:
    if [ $? -ne 0 ]; then
        echo_error "التثبيت بواسطة pkexec فشل أو تم إلغاؤه."
        exit 1
    fi
    echo_info "التثبيت اكتمل، الآن تشغيل الواجهة الرسومية..."
    # شغل واجهة البايثون كـ user عادي
    python3 "$SCRIPT_DIR/video_editor_gui.py"
    exit $?
fi

# إذا وصلنا هنا والوسيط هو --install، فنحن تحت صلاحيات root: نقوم بالتثبيت ثم ننهي
if [ "$1" == "--install" ]; then
    echo_info "مرحلة التثبيت بصلاحيات root..."
    # تحديث قائمة الحزم وتثبيت المستودعات والحزم المطلوبة
    # 1. تمكين universe repository
    ensure_add_apt_repository_available
    echo_info "تمكين universe repository (إذا لم يكن مفعلًا)"
    add-apt-repository -y universe
    echo_info "تحديث قائمة الحزم"
    apt-get update -y

    # 2. تثبيت python3-tk
    echo_info "التحقق وتثبيت python3-tk"
    apt_install_if_missing python3-tk

    # 3. تثبيت ffmpeg
    echo_info "التحقق وتثبيت ffmpeg"
    apt_install_if_missing ffmpeg

    # 4. تثبيت pip3 إن لم يكن موجودًا (لا بد بعد python3-tk، لكن غالبًا python3-pip غير مثبت افتراضي)
    if ! command -v pip3 >/dev/null 2>&1; then
        echo_info "تثبيت python3-pip"
        apt-get install -y python3-pip
    else
        echo_info "pip3 موجود"
    fi

    # 5. تثبيت مكتبات بايثون المطلوبة عبر pip3 (يمكن تعديل القائمة حسب احتياجات video_editor_gui.py)
    echo_info "التحقق من مكتبات بايثون وتنصيب المفقود"
    MISSING_PY_LIBS=()
    check_py_module() {
        python3 - <<EOF
import sys
try:
    import $1
except Exception:
    sys.exit(1)
sys.exit(0)
EOF
        if [ $? -ne 0 ]; then
            MISSING_PY_LIBS+=("$1")
        fi
    }
    # مثال: قائمة المكتبات المحتملة
    check_py_module cv2
    check_py_module numpy
    check_py_module PIL
    check_py_module pydub
    check_py_module imageio_ffmpeg
    check_py_module matplotlib
    check_py_module tkinter

    if [ ${#MISSING_PY_LIBS[@]} -ne 0 ]; then
        echo_info "مكتبات بايثون مفقودة: ${MISSING_PY_LIBS[*]}"
        for mod in "${MISSING_PY_LIBS[@]}"; do
            case "$mod" in
                cv2) PKG_NAME="opencv-python" ;;
                PIL) PKG_NAME="Pillow" ;;
                pydub) PKG_NAME="pydub" ;;
                imageio_ffmpeg) PKG_NAME="imageio-ffmpeg" ;;
                matplotlib) PKG_NAME="matplotlib" ;;
                numpy) PKG_NAME="numpy" ;;
                tkinter)
                    # tkinter مع python3-tk، لقد ثبتناه سابقًا
                    continue
                    ;;
                *)
                    PKG_NAME="$mod"
                    ;;
            esac
            echo_info "تثبيت $PKG_NAME عبر pip3"
            pip3 install "$PKG_NAME"
            if [ $? -ne 0 ]; then
                echo_error "فشل تثبيت $PKG_NAME"
                # نتابع رغم الخطأ أو ننهي؟ يمكن هنا إنهاء:
                # exit 1
            fi
        done
    else
        echo_info "جميع مكتبات بايثون المطلوبة مثبتة"
    fi

    echo_info "انتهت مرحلة التثبيت."
    exit 0
fi

# إذا وصلنا هنا: المستخدم عادي (EUID!=0) بعد الانتهاء من تثبيت --install، أو ربما EUID==0 مباشرة بدون --install
# في حالة EUID==0 دون --install فهذا غير متوقع؛ لكن نتعامل معه بتشغيل التثبيت مباشرة:
if [ "$EUID" -eq 0 ]; then
    echo_info "تم التشغيل بصلاحيات root بدون --install، نقوم بالتثبيت ثم ننهي."
    # ننفّذ نفس كود التثبيت:
    ensure_add_apt_repository_available
    echo_info "تمكين universe repository"
    add-apt-repository -y universe
    echo_info "تحديث"
    apt-get update -y
    apt_install_if_missing python3-tk
    apt_install_if_missing ffmpeg
    if ! command -v pip3 >/dev/null 2>&1; then
        echo_info "تثبيت python3-pip"
        apt-get install -y python3-pip
    fi
    # تثبيت مكتبات بايثون كما أعلاه (يمكن إعادة استخدام الدالة)
    # ...
    echo_info "انتهى التثبيت كـ root. الآن إنهاء حتى يعيد المستخدم تشغيله كـ user."
    exit 0
fi

# في الوضع الطبيعي: تشغيل الواجهة الرسومية
echo_info "تشغيل سكربت البايثون الرسومي"
python3 "$SCRIPT_DIR/video_editor_gui.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo_error "انتهى سكربت البايثون برمز خروج: $EXIT_CODE"
    exit $EXIT_CODE
fi
echo_info "انتهى بنجاح."
exit 0
