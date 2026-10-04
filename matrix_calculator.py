import streamlit as st
import numpy as np

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Kalkulator matrix",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS / TAMPILAN
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 80% 10%, #172554 0%, transparent 35%),
        linear-gradient(135deg, #070d1a, #0d1729 55%, #111827);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #080d18);
    border-right: 1px solid #263650;
}

.sidebar-title {
    font-size: 27px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 25px;
}

.sidebar-text {
    color: #9fb3cf;
    line-height: 1.7;
}

/* JUDUL */

.main-title {
    font-size: 54px;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
    margin-top: 4px;
    margin-bottom: 25px;
}

/* CARD */

.matrix-card {
    background: rgba(17, 30, 52, 0.90);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #29456e;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(
        135deg,
        #082f49,
        #172554
    );
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #2187b8;
    box-shadow: 0 10px 35px rgba(0,0,0,0.3);
}

/* JUDUL MATRIKS */

.matrix-title {
    color: #38bdf8;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 15px;
}

/* TOMBOL */

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    border: 1px solid #334d73;
    background: #111827;
    color: #e2e8f0;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
    transform: translateY(-2px);
}

/* TOMBOL HITUNG */

.calculate-button {
    background: linear-gradient(
        90deg,
        #0284c7,
        #4f46e5
    );
    color: white;
    padding: 15px;
    border-radius: 14px;
    text-align: center;
    font-size: 18px;
    font-weight: 800;
    margin-top: 20px;
}

/* INPUT */

div[data-testid="stNumberInput"] input {
    background-color: #111827;
    color: white;
    border: 1px solid #334d73;
    border-radius: 10px;
}

/* SELECTBOX */

div[data-baseweb="select"] > div {
    background-color: #111827;
    border-radius: 10px;
    border: 1px solid #334d73;
}

/* GARIS */

hr {
    border-color: #293b57;
}

/* TEKS */

h1, h2, h3 {
    color: #f8fafc;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI MATRIKS
# ============================================================

def get_matrix(prefix, rows, cols):
    matrix = []

    for i in range(rows):
        row = []

        for j in range(cols):
            value = st.number_input(
                f"{prefix}[{i+1},{j+1}]",
                value=0.0,
                step=1.0,
                key=f"{prefix}_{i}_{j}"
            )
            row.append(value)

        matrix.append(row)

    return np.array(matrix, dtype=float)


def format_matrix(matrix):
    return np.round(matrix, 4)


def rref(matrix):
    A = matrix.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0
    steps = []

    for col in range(cols):

        if pivot_row >= rows:
            break

        max_row = pivot_row + np.argmax(
            np.abs(A[pivot_row:, col])
        )

        if abs(A[max_row, col]) < 1e-10:
            continue

        if max_row != pivot_row:
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
            steps.append(
                f"Tukar baris R{pivot_row+1} ↔ R{max_row+1}"
            )

        pivot = A[pivot_row, col]

        if abs(pivot - 1) > 1e-10:
            A[pivot_row] = A[pivot_row] / pivot
            steps.append(
                f"R{pivot_row+1} → R{pivot_row+1} / {pivot:.4g}"
            )

        for row in range(rows):

            if row != pivot_row:
                factor = A[row, col]

                if abs(factor) > 1e-10:
                    A[row] = A[row] - factor * A[pivot_row]

                    steps.append(
                        f"R{row+1} → R{row+1} - "
                        f"({factor:.4g})R{pivot_row+1}"
                    )

        pivot_row += 1

    return A, steps


def adjoint_matrix(A):
    n, m = A.shape

    if n != m:
        raise ValueError("Adjoin hanya dapat digunakan pada matriks persegi.")

    if n == 1:
        return np.array([[1.0]])

    cofactors = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            minor = np.delete(
                np.delete(A, i, axis=0),
                j,
                axis=1
            )

            cofactors[i, j] = (
                (-1) ** (i + j)
            ) * np.linalg.det(minor)

    return cofactors.T


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">⚙ MATRIXLAB</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Pilih Operasi")

    operation = st.selectbox(
        "Operasi",
        [
            "Penjumlahan",
            "Pengurangan",
            "Perkalian",
            "Transpose",
            "Determinan",
            "Invers",
            "Adjoin",
            "RREF / Gauss-Jordan"
        ],
        label_visibility="collapsed"
    )

    st.markdown("### Ukuran Matriks")

    size = st.selectbox(
        "Ukuran",
        ["2 × 2", "3 × 3", "4 × 4"],
        label_visibility="collapsed"
    )

    n = int(size[0])

    st.divider()

    st.markdown("### Tentang MatrixLab")

    st.markdown(
        """
        <div class="sidebar-text">
        Kalkulator matriks interaktif yang dibuat
        menggunakan <b>Python</b>, <b>NumPy</b>,
        dan <b>Streamlit</b>.
        <br><br>
        <b>Operasi tersedia:</b>
        <br>• Penjumlahan
        <br>• Pengurangan
        <br>• Perkalian
        <br>• Transpose
        <br>• Determinan
        <br>• Invers
        <br>• Adjoin
        <br>• RREF / Gauss-Jordan
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">∑ MATRIXLAB</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Kalkulator Matriks Interaktif Berbasis Python'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOMBOL CONTOH
# ============================================================

col1, col2 = st.columns(2)

with col1:
    contoh = st.button(
        "💡 Gunakan Contoh",
        use_container_width=True
    )

with col2:
    reset = st.button(
        "🗑️ Atur Ulang",
        use_container_width=True
    )


# ============================================================
# CONTOH / RESET
# ============================================================

if contoh:

    contoh_A = np.arange(1, n*n + 1).reshape(n, n)
    contoh_B = np.arange(9, 9 + n*n).reshape(n, n)

    for i in range(n):
        for j in range(n):

            st.session_state[
                f"A_{i}_{j}"
            ] = float(contoh_A[i, j])

            st.session_state[
                f"B_{i}_{j}"
            ] = float(contoh_B[i, j])


if reset:

    for i in range(n):
        for j in range(n):

            st.session_state[
                f"A_{i}_{j}"
            ] = 0.0

            st.session_state[
                f"B_{i}_{j}"
            ] = 0.0


st.divider()


# ============================================================
# INPUT MATRIKS
# ============================================================

need_B = operation in [
    "Penjumlahan",
    "Pengurangan",
    "Perkalian"
]


if need_B:

    colA, colB = st.columns(2)

    with colA:

        st.markdown(
            '<div class="matrix-title">Matriks A</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="matrix-card">',
            unsafe_allow_html=True
        )

        A = get_matrix("A", n, n)

        st.markdown("</div>", unsafe_allow_html=True)

    with colB:

        st.markdown(
            '<div class="matrix-title">Matriks B</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="matrix-card">',
            unsafe_allow_html=True
        )

        B = get_matrix("B", n, n)

        st.markdown("</div>", unsafe_allow_html=True)

else:

    st.markdown(
        '<div class="matrix-title">Matriks A</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="matrix-card">',
        unsafe_allow_html=True
    )

    A = get_matrix("A", n, n)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TOMBOL HITUNG
# ============================================================

st.markdown(
    '<div class="calculate-button">'
    '🚀 SIAP MENGHITUNG MATRIKS'
    '</div>',
    unsafe_allow_html=True
)

hitung = st.button(
    "HITUNG MATRIKS",
    use_container_width=True
)


# ============================================================
# PROSES PERHITUNGAN
# ============================================================

if hitung:

    try:

        result = None
        title = ""

        # ----------------------------------------------------
        # PENJUMLAHAN
        # ----------------------------------------------------

        if operation == "Penjumlahan":

            result = A + B
            title = "Hasil A + B"

        # ----------------------------------------------------
        # PENGURANGAN
        # ----------------------------------------------------

        elif operation == "Pengurangan":

            result = A - B
            title = "Hasil A - B"

        # ----------------------------------------------------
        # PERKALIAN
        # ----------------------------------------------------

        elif operation == "Perkalian":

            result = A @ B
            title = "Hasil A × B"

        # ----------------------------------------------------
        # TRANSPOSE
        # ----------------------------------------------------

        elif operation == "Transpose":

            result = A.T
            title = "Transpose Matriks A"

        # ----------------------------------------------------
        # DETERMINAN
        # ----------------------------------------------------

        elif operation == "Determinan":

            result = np.linalg.det(A)
            title = "Determinan Matriks A"

        # ----------------------------------------------------
        # INVERS
        # ----------------------------------------------------

        elif operation == "Invers":

            det = np.linalg.det(A)

            if abs(det) < 1e-10:

                st.error(
                    "Matriks tidak mempunyai invers "
                    "karena determinannya = 0."
                )

                st.stop()

            result = np.linalg.inv(A)
            title = "Invers Matriks A"

        # ----------------------------------------------------
        # ADJOIN
        # ----------------------------------------------------

        elif operation == "Adjoin":

            det = np.linalg.det(A)

            if abs(det) < 1e-10:

                st.warning(
                    "Determinan matriks = 0. "
                    "Adjoin tetap dapat dihitung, "
                    "tetapi matriks tidak mempunyai invers."
                )

            result = adjoint_matrix(A)
            title = "Adjoin Matriks A"

        # ----------------------------------------------------
        # RREF
        # ----------------------------------------------------

        elif operation == "RREF / Gauss-Jordan":

            result, steps = rref(A)

            title = "RREF / Bentuk Eselon Baris Tereduksi"

        # ----------------------------------------------------
        # TAMPILKAN HASIL
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"## 📊 {title}"
        )

        if operation == "Determinan":

            st.success(
                f"Nilai determinan = "
                f"**{result:.6f}**"
            )

        else:

            st.dataframe(
                format_matrix(result),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # LANGKAH RREF
        # ----------------------------------------------------

        if operation == "RREF / Gauss-Jordan":

            st.subheader("🧮 Langkah Gauss-Jordan")

            if len(steps) == 0:

                st.info(
                    "Tidak ada operasi baris yang diperlukan."
                )

            else:

                for i, step in enumerate(steps, 1):

                    st.write(
                        f"**Langkah {i}:** {step}"
                    )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan dalam perhitungan: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MatrixLab • Dibuat menggunakan Python + NumPy + Streamlit"
)
