import statistics
import time
import sqlite3
from dataclasses import dataclass, replace
from typing import Dict, List, Optional
import streamlit as st

# ============================================================================
# 1. DATA ACCESS LAYER (THE MODEL)
# ============================================================================
# Responsible ONLY for storing/retrieving patient records and performing
# basic data-integrity cleanup on load. Contains NO clinical decision logic
# and NO presentation/I/O logic.
# ============================================================================

@dataclass
class Patient:
    """Plain data structure representing a patient's baseline metrics."""
    patient_id: int
    glucose: float
    bmi: float
    age: int
    blood_pressure: float


class PatientRepository:
    """
    Data Access Layer.

    Owns the SQLite 'database' of patients, seeds mock data, and applies
    initialization-time cleanup rules (e.g. fixing missing BMI values).
    """

    def __init__(self, db_name: str = "patients.db") -> None:
        self.db_name = db_name
        self._conn = None
        self._connect()
        self._create_table()
        # Only seed data if the table is empty (first run)
        if not self.get_all_ids():
            self._seed_mock_data()
            self._clean_missing_bmi()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self.db_name, check_same_thread=False) # Allow multiple threads for Streamlit
        self._conn.row_factory = sqlite3.Row # Access columns by name

    def _create_table(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY,
                glucose REAL,
                bmi REAL,
                age INTEGER,
                blood_pressure REAL
            )
        """)
        self._conn.commit()

    # -- Seeding -------------------------------------------------------------
    def _seed_mock_data(self) -> None:
        """Loads 4 sample patients into the database."""
        mock_records = [
            Patient(patient_id=1, glucose=95, bmi=22.5, age=28, blood_pressure=115),
            Patient(patient_id=2, glucose=130, bmi=31.0, age=52, blood_pressure=142),
            Patient(patient_id=3, glucose=110, bmi=0.0, age=61, blood_pressure=128),  # BMI missing
            Patient(patient_id=4, glucose=180, bmi=27.8, age=45, blood_pressure=150),
        ]
        cursor = self._conn.cursor()
        for p in mock_records:
            cursor.execute("""
                INSERT INTO patients (patient_id, glucose, bmi, age, blood_pressure)
                VALUES (?, ?, ?, ?, ?)
            """, (p.patient_id, p.glucose, p.bmi, p.age, p.blood_pressure))
        self._conn.commit()

    # -- Cleanup --------------------------------------------------------------
    def _clean_missing_bmi(self) -> None:
        """
        Data cleanup rule: any patient with BMI == 0 is treated as missing
        data. It is replaced with the median BMI computed from all OTHER
        patients with a valid (non-zero) BMI.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT bmi FROM patients WHERE bmi != 0")
        valid_bmis_rows = cursor.fetchall()
        valid_bmis = [row['bmi'] for row in valid_bmis_rows]

        if not valid_bmis:
            return  # No valid BMI values to compute a median from.

        median_bmi = statistics.median(valid_bmis)

        cursor.execute("UPDATE patients SET bmi = ? WHERE bmi = 0", (median_bmi,))
        self._conn.commit()

    # -- Public data-access API ------------------------------------------------
    def get_all_ids(self) -> List[int]:
        """Returns a sorted list of all valid patient IDs."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT patient_id FROM patients ORDER BY patient_id")
        return [row['patient_id'] for row in cursor.fetchall()]

    def get_patient(self, patient_id: int) -> Optional[Patient]:
        """Fetches a single patient record by ID, or None if not found."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        if row:
            return Patient(
                patient_id=row['patient_id'],
                glucose=row['glucose'],
                bmi=row['bmi'],
                age=row['age'],
                blood_pressure=row['blood_pressure']
            )
        return None

    def update_patient(self, patient_id: int, **updated_fields) -> Optional[Patient]:
        """Persists updated metric values for an existing patient."""
        existing_patient = self.get_patient(patient_id)
        if existing_patient is None:
            return None

        # Build update query dynamically
        set_clauses = []
        values = []
        for field, value in updated_fields.items():
            set_clauses.append(f"{field} = ?")
            values.append(value)

        if not set_clauses:
            return existing_patient # Nothing to update

        values.append(patient_id)
        query = f"UPDATE patients SET {', '.join(set_clauses)} WHERE patient_id = ?"

        cursor = self._conn.cursor()
        cursor.execute(query, tuple(values))
        self._conn.commit()
        return self.get_patient(patient_id) # Return the updated patient

    def close(self) -> None:
        """Closes the database connection."""
        if self._conn:
            self._conn.close()


# ============================================================================
# 2. BUSINESS LOGIC LAYER (THE SERVICE)
# ============================================================================
# Pure clinical decision rules. No print statements, no input() calls, no
# knowledge of the repository or the console — just data in, data out.
# This makes the layer trivially unit-testable.
# ============================================================================

class RiskScoringService:
    """
    Business Logic Layer.

    Encapsulates the clinical scoring rules used to evaluate diabetes risk.
    Each metric is scored 0 (Low), 1 (Medium), or 2 (High); the four scores
    are summed into a total risk score which maps to a risk category.
    """

    # Category score bands
    LOW_RISK_MAX = 2       # 0-2  => Low Risk
    MODERATE_RISK_MAX = 5  # 3-5  => Moderate Risk
    # 6+ => High Risk

    @staticmethod
    def score_glucose(glucose: float) -> int:
        """Glucose (mg/dL): <100 Low, 100-125 Medium, >=126 High."""
        if glucose < 100:
            return 0
        elif glucose <= 125:
            return 1
        else:
            return 2

    @staticmethod
    def score_bmi(bmi: float) -> int:
        """BMI: <25 Low, 25-29.9 Medium, >=30 High."""
        if bmi < 25:
            return 0
        elif bmi < 30:
            return 1
        else:
            return 2

    @staticmethod
    def score_age(age: int) -> int:
        """Age (years): <40 Low, 40-59 Medium, >=60 High."""
        if age < 40:
            return 0
        elif age < 60:
            return 1
        else:
            return 2

    @staticmethod
    def score_blood_pressure(blood_pressure: float) -> int:
        """Systolic Blood Pressure (mmHg): <120 Low, 120-139 Medium, >=140 High."""
        if blood_pressure < 120:
            return 0
        elif blood_pressure < 140:
            return 1
        else:
            return 2

    @classmethod
    def calculate_total_score(cls, patient: Patient) -> int:
        """Sums the individual metric scores into one total risk score (0-8)."""
        return (
            cls.score_glucose(patient.glucose)
            + cls.score_bmi(patient.bmi)
            + cls.score_age(patient.age)
            + cls.score_blood_pressure(patient.blood_pressure)
        )

    @classmethod
    def categorize_risk(cls, total_score: int) -> str:
        """Maps a total score to a human-readable risk category."""
        if total_score <= cls.LOW_RISK_MAX:
            return "Low Risk"
        elif total_score <= cls.MODERATE_RISK_MAX:
            return "Moderate Risk"
        else:
            return "High Risk"

    @classmethod
    def evaluate(cls, patient: Patient) -> dict:
        """
        Runs the full evaluation pipeline for a patient and returns a
        structured breakdown suitable for reporting.
        """
        breakdown = {
            "Glucose": cls.score_glucose(patient.glucose),
            "BMI": cls.score_bmi(patient.bmi),
            "Age": cls.score_age(patient.age),
            "BloodPressure": cls.score_blood_pressure(patient.blood_pressure),
        }
        total = sum(breakdown.values())
        category = cls.categorize_risk(total)
        return {"breakdown": breakdown, "total_score": total, "category": category}


# ============================================================================
# APPLICATION ENTRYPOINT for Streamlit
# ============================================================================

def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Scoring System", layout="centered")
    st.title("Diabetes Risk Scoring System")
    st.markdown("---\n" * 1)

    # Initialize repository and service, storing repository in session state for persistence
    if 'repository' not in st.session_state:
        st.session_state.repository = PatientRepository()
    repository = st.session_state.repository
    service = RiskScoringService()

    # Initialize other session state variables
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'patient_metrics' not in st.session_state:
        st.session_state.patient_metrics = {}
    if 'report_content' not in st.session_state:
        st.session_state.report_content = ""

    # --- Patient Selection ---
    st.subheader("Patient Selection")
    patient_ids = repository.get_all_ids()
    # Ensure the selected_patient_id is still valid if the list of patients changes
    if 'selected_patient_id' not in st.session_state or st.session_state.selected_patient_id not in patient_ids:
        st.session_state.selected_patient_id = patient_ids[0] if patient_ids else None

    selected_patient_id_display = st.selectbox(
        "Select a Patient ID:",
        options=patient_ids,
        index=patient_ids.index(st.session_state.selected_patient_id) if st.session_state.selected_patient_id in patient_ids else (0 if patient_ids else None),
        key="select_patient_id_widget"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Load Patient Profile", key="load_button"):
            if selected_patient_id_display:
                st.session_state.selected_patient_id = selected_patient_id_display
                patient = repository.get_patient(st.session_state.selected_patient_id)
                if patient:
                    st.session_state.current_patient = patient
                    st.session_state.patient_metrics = {
                        "glucose": patient.glucose,
                        "bmi": patient.bmi,
                        "age": patient.age,
                        "blood_pressure": patient.blood_pressure,
                    }
                    st.session_state.report_content = ""
                    st.success(f"Patient {st.session_state.selected_patient_id} loaded.")
                else:
                    st.error(f"Patient ID {st.session_state.selected_patient_id} not found.")
            else:
                st.warning("Please select a Patient ID to load.")

    # --- Clinical Metrics Input ---
    st.subheader("Clinical Metrics")
    if st.session_state.current_patient:
        current_patient_data = st.session_state.current_patient
        st.write(f"**Patient ID: {current_patient_data.patient_id}**")

        # Use session_state.patient_metrics to pre-fill inputs
        glucose_val = st.session_state.patient_metrics.get("glucose", current_patient_data.glucose)
        bmi_val = st.session_state.patient_metrics.get("bmi", current_patient_data.bmi)
        age_val = st.session_state.patient_metrics.get("age", current_patient_data.age)
        blood_pressure_val = st.session_state.patient_metrics.get("blood_pressure", current_patient_data.blood_pressure)

        st.session_state.patient_metrics["glucose"] = st.number_input("Glucose (mg/dL)", value=float(glucose_val), key="input_glucose")
        st.session_state.patient_metrics["bmi"] = st.number_input("BMI", value=float(bmi_val), key="input_bmi")
        st.session_state.patient_metrics["age"] = st.number_input("Age (years)", value=int(age_val), key="input_age")
        st.session_state.patient_metrics["blood_pressure"] = st.number_input("Blood Pressure (mmHg)", value=float(blood_pressure_val), key="input_blood_pressure")

        if st.button("Calculate & Assess Risk", key="assess_button"):
            try:
                updated_glucose = st.session_state.patient_metrics["glucose"]
                updated_bmi = st.session_state.patient_metrics["bmi"]
                updated_age = st.session_state.patient_metrics["age"]
                updated_blood_pressure = st.session_state.patient_metrics["blood_pressure"]

                # Update the repository's patient data first
                updated_patient = repository.update_patient(
                    current_patient_data.patient_id,
                    glucose=updated_glucose,
                    bmi=updated_bmi,
                    age=updated_age,
                    blood_pressure=updated_blood_pressure
                )
                
                if updated_patient:
                    st.session_state.current_patient = updated_patient # Update current patient in session state
                    evaluation = service.evaluate(updated_patient)

                    border = "*" * 42
                    report_lines = [
                        f"```markdown",
                        f"# {border}",
                        f"#      OFFICIAL MEDICAL DIAGNOSTIC REPORT",
                        f"# {border}",
                        f"#  Patient ID:       {updated_patient.patient_id}",
                        f"#  Cumulative Score: {evaluation['total_score']} pts",
                        f"#  Risk Category:    {evaluation['category'].upper()}",
                        f"# {border}",
                        f"#  *** CONFIDENTIAL - MEDICAL RECORD *** ",
                        f"# {border}",
                        f"```"
                    ]
                    st.session_state.report_content = "\n".join(report_lines)
                    st.success("Patient data updated and risk assessed!")
                else:
                    st.error(f"Failed to update patient {current_patient_data.patient_id}.")

            except ValueError:
                st.error("Please ensure all metrics are valid numeric values.")

    else:
        st.info("Load a patient profile to view and edit clinical metrics.")

    # --- Diagnostic Report ---
    if st.session_state.report_content:
        st.subheader("Diagnostic Risk Report")
        st.markdown(st.session_state.report_content)


if __name__ == "__main__":
    main()
