import os
import requests
import pandas as pd
from abc import ABC, abstractmethod

# ======================================================
# =============== CLASE BASE ===========================
# ======================================================

class MoffinBuroBase(ABC):
    def __init__(self, rfc: str, service_name: str, original_key_prefix: str):
        self.rfc = rfc
        self.service_name = service_name
        self.original_key_prefix = original_key_prefix

        self.base_url = "https://app.moffin.mx/api/v1"
        self.token = os.getenv("MOFFIN_TOKEN", "").strip()

        if not self.token:
            raise EnvironmentError("MOFFIN_TOKEN no definido")

        self.headers = {
            "Authorization": f"Token {self.token}",
            "Accept": "application/json",
        }

        self._bureau_json: dict | None = None
        self._fecha_consulta: str | None = None
        self._original_key: str | None = None

    # ------------------------------
    # Obtener JSON más reciente
    # ------------------------------
    def _obtener_json_mas_reciente(self, limit: int = 50) -> None:
        offset = 0

        while True:
            params = {
                "search": self.rfc,
                "limit": limit,
                "offset": offset,
                "order": "DESC",
            }

            resp = requests.get(
                f"{self.base_url}/service_queries",
                headers=self.headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            batch = data.get("serviceQueries", [])
            if not batch:
                break

            for q in batch:
                if q.get("service") == self.service_name and q.get("response"):
                    self._bureau_json = q["response"]
                    self._fecha_consulta = q.get("createdAt")
                    self._original_key = (
                        f"{self.original_key_prefix}/{self.rfc}/{self._fecha_consulta}"
                        if self._fecha_consulta
                        else f"{self.original_key_prefix}/{self.rfc}"
                    )
                    return

            if len(batch) < limit:
                break

            offset += limit

        raise ValueError(f"No se encontró {self.service_name} para el RFC {self.rfc}")

    # ------------------------------
    # Métodos a implementar
    # ------------------------------
    @abstractmethod
    def _extraer_registros(self) -> list[dict]:
        pass

    # ------------------------------
    # Normalización base
    # ------------------------------
    def _estructurar_dataframe(self, registros: list[dict]) -> pd.DataFrame:
        if not registros:
            return pd.DataFrame()

        return pd.json_normalize(registros)

    # ------------------------------
    # Hook de formato final
    # ------------------------------
    def formatear_tabla(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
    def _formatear_fecha_consulta(self):
        """
        Convierte createdAt tipo:
        '2024-06-10T11:28:08.123Z' -> '2024-06-10'
        """
        try:
            if not self._fecha_consulta:
                print("no hay fecha de consulta")
                return None
                
            return self._fecha_consulta.split("T")[0]
        except Exception:
            return None

    # ------------------------------
    # Orquestador
    # ------------------------------
    def caller(self) -> pd.DataFrame:
        self._obtener_json_mas_reciente()
        registros = self._extraer_registros()
        df = self._estructurar_dataframe(registros)

        # 👇 AQUÍ inyectamos la fecha real de consulta
        if not df.empty:
            df["Fecha consulta"] = self._formatear_fecha_consulta()

        return self.formatear_tabla(df)

# ======================================================
# =============== PERSONA FÍSICA =======================
# ======================================================

class BuroMoffinPF(MoffinBuroBase):
    def __init__(self, rfc: str):
        super().__init__(
            rfc=rfc,
            service_name="bureau_pf",
            original_key_prefix="moffin_pf",
        )

    def _extraer_registros(self) -> list[dict]:
        persona = self._bureau_json["return"]["Personas"]["Persona"][0]
        cuentas = persona.get("Cuentas", {}).get("Cuenta", [])

        if isinstance(cuentas, dict):
            cuentas = [cuentas]

        return cuentas or []

    # -------- helpers internos --------
    def _formatear_fecha(self, valor):
        """
        Convierte '28102025' -> '2025-10-28'
        """
        if not isinstance(valor, str) or len(valor) != 8:
            return None
        try:
            return f"{valor[4:8]}-{valor[2:4]}-{valor[0:2]}"
        except Exception:
            return None
    
    def _obtener_monto_pagar(self, df: pd.DataFrame) -> float:
        """
        Calcula el monto total a pagar a partir de la columna MontoPagar.
        Regresa float (no formateado).
        """
        if "MontoPagar" not in df.columns:
            return 0.0

        try:
            montos = (
                df["MontoPagar"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.strip()
                .replace("", "0")
                .astype(float)
            )
            return montos.sum()
        except Exception:
            return 0.0


    def _formatear_monto(self, valor):
        """
        Convierte números o strings a formato dinero: $12,345.67
        Limpia símbolos como '$' y separadores de miles.
        """
        try:
            if valor is None:
                return None

            # Si es string, limpiamos símbolos comunes
            if isinstance(valor, str):
                valor = (
                    valor
                    .replace("$", "")
                    .replace(",", "")
                    .replace("+", "")
                    .strip()
                )

                if valor == "":
                    return None

            monto = float(valor)
            return f"${monto:,.2f}"

        except Exception:
            return None


    def formatear_tabla(self, df: pd.DataFrame) -> pd.DataFrame:
        columnas = {
            "FechaActualizacion": "Fecha actualización",
            "FechaAperturaCuenta": "Fecha apertura",
            "NombreOtorgante": "Otorgante",
            "TipoCuenta": "Tipo de cuenta",
            "TipoContrato": "Tipo de contrato",
            "FrecuenciaPagos": "Frecuencia de pago",
            "MontoPagar": "Monto a pagar",
            "SaldoActual":"Saldo Actual",
            "Fecha consulta": "Fecha Consulta",
            "HistoricoPagos":"Comportamiento"


        }

        # Seleccionar solo columnas relevantes
        df_out = df[[c for c in columnas if c in df.columns]].copy()

        # Formatear fechas
        for col in ["FechaActualizacion", "FechaAperturaCuenta"]:
            if col in df_out.columns:
                df_out[col] = df_out[col].apply(self._formatear_fecha)

        # Formatear monto a pagar
        if "MontoPagar" in df_out.columns:
            df_out["MontoPagar"] = df_out["MontoPagar"].apply(self._formatear_monto)

        if "SaldoActual" in df_out.columns:
            df_out["SaldoActual"] = df_out["SaldoActual"].apply(self._formatear_monto)



        # Motnto Total a pagar
        monto_total = self._obtener_monto_pagar(df_out)
        df_out["MontoTotalPagar"] = monto_total
        df_out["MontoTotalPagar"] = df_out["MontoTotalPagar"].apply(self._formatear_monto)

        # Renombrar columnas
        df_out.rename(columns=columnas, inplace=True)

        # Orden lógico
        if "Fecha apertura" in df_out.columns:
            df_out.sort_values(
                by="Fecha apertura",
                ascending=False,
                inplace=True,
                ignore_index=True,
            )

    
            
        return df_out.drop(columns = "original_key", errors="ignore")
    # ======================================================
    # =============== PERSONA MORAL ========================
    # ======================================================


# ======================================================
# =============== PERSONA MORAl ========================
# ======================================================

class BuroMoffinPM(MoffinBuroBase):
    def __init__(self, rfc: str):
        super().__init__(
            rfc=rfc,
            service_name="bureau_pm",
            original_key_prefix="moffin_pm",
        )

    def _extraer_registros(self) -> list[dict]:
        respuesta = self._bureau_json.get("respuesta", {})
        credito = respuesta.get("creditoFinanciero", [])

        if isinstance(credito, dict):
            credito = [credito]

        return credito or []



# ======================================================
# =============== FUNCIÓN ÚNICA DE ENTRADA ==============
# ======================================================

def obtener_buro_moffin_por_rfc(rfc: str) -> pd.DataFrame:
    if not isinstance(rfc, str):
        raise TypeError("El RFC debe ser string")

    rfc = rfc.strip().upper()

    if len(rfc) == 13:
        return BuroMoffinPF(rfc).caller()

    elif len(rfc) == 12:
        df = BuroMoffinPM(rfc).caller()
        if "Fecha Consulta" not in df.columns:
            df = df.rename(columns = {"Fecha consulta":"Fecha Consulta"})
        return df
    else:
        raise ValueError(
            f"RFC inválido: '{rfc}'. PF = 13 caracteres, PM = 12."
        )
    
print(obtener_buro_moffin_por_rfc("AAG230203211"))


