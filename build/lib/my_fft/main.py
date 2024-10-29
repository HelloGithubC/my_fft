import numpy as np
import os, joblib,sys
from scipy.fft import rfftn, irfftn
from .lib.fftpower import FFTPower_CPP
from .lib.mesh import ToMesh


class FFTPowerCPP:
    def __init__(self, Nmesh, BoxSize, shotnoise=0.0):
        if isinstance(Nmesh, int) or isinstance(Nmesh, float):
            self.Nmesh = np.array([Nmesh, Nmesh, Nmesh], dtype=np.int32)
        else:
            self.Nmesh = np.array(Nmesh, dtype=np.int32)
        if isinstance(BoxSize, float) or isinstance(BoxSize, int):
            self.BoxSize = np.array([BoxSize, BoxSize, BoxSize], dtype=float)
        else:
            self.BoxSize = np.array(BoxSize, dtype=float)

        self.attrs = {
            "Nmesh": self.Nmesh,
            "BoxSize": self.BoxSize,
            "shotnoise": shotnoise,
        }
        self.power = None
        self.fft = None
        self.is_deal = False
        self.done_conj = False

    def run(
        self,
        field,
        kmin,
        kmax,
        dk,
        Nmu=None,
        k_arrays=None,
        mode="1d",
        field_type="complex",
        right=False,
        linear=True,
        force_conj = False,
        nthreads=1,
    ):
        do_conj = True if force_conj else (not self.done_conj)
        self.fft = FFTPower_CPP(self.BoxSize)
        self.attrs["kmin"] = kmin
        self.attrs["kmax"] = kmax
        if dk < 0:
            dk = 2 * np.pi / self.attrs["BoxSize"]
        self.attrs["dk"] = dk
        k_array = np.arange(kmin, kmax, dk)
        self.attrs["Nk"] = len(k_array) - 1
        self.attrs["mode"] = mode
        if mode == "2d":
            if not isinstance(Nmu, int):
                raise ValueError("Nmu must be an integer")
            else:
                self.attrs["Nmu"] = Nmu
                mu_array = np.linspace(0, 1, Nmu + 1, endpoint=True)
        else:
            self.attrs["Nmu"] = 1
            mu_array = np.array([0.0, 1.0])

        if field_type == "real":
            field_complex = rfftn(field) / self.attrs["Nmesh"].astype(float).prod()
        else:
            field_complex = field

        if k_arrays is None:
            k_x_array = (
                np.fft.fftfreq(self.Nmesh[0], d=1.0)
                * 2.0
                * np.pi
                * self.Nmesh[0]
                / self.BoxSize[0]
            )
            k_y_array = (
                np.fft.fftfreq(self.Nmesh[1], d=1.0)
                * 2.0
                * np.pi
                * self.Nmesh[1]
                / self.BoxSize[1]
            )
            k_z_array = (
                np.fft.fftfreq(self.Nmesh[2], d=1.0)
                * 2.0
                * np.pi
                * self.Nmesh[2]
                / self.BoxSize[2]
            )[: field_complex.shape[2]]
        else:
            k_x_array, k_y_array, k_z_array = k_arrays

        power = np.zeros((self.attrs["Nk"], self.attrs["Nmu"]), dtype=np.complex128)
        power_mu = np.zeros_like(power, dtype=np.float64)
        power_k = np.zeros_like(power, dtype=np.float64)
        power_modes = np.zeros_like(power, dtype=np.int32)
        self.fft.RunFromComplex(
            power,
            power_mu,
            power_k,
            power_modes,
            field_complex,
            k_array,
            mu_array,
            kmin,
            kmax,
            k_x_array,
            k_y_array,
            k_z_array,
            mode,
            right,
            linear=linear,
            do_conj=do_conj,
            nthreads=nthreads,
        )
        self.done_conj = True
        power_k[power_modes == 0] = np.nan
        if mode == "2d":
            power_mu[power_modes == 0] = np.nan
        power[power_modes == 0] = np.nan
        if mode == "2d":
            self.power = {"k": power_k, "mu": power_mu, "Pkmu": power, "modes": power_modes}
        else:
            self.power = {"k": power_k, "Pk": power, "modes": power_modes}
        self.attrs["Nmu"] = Nmu
        self.attrs["kmin"] = kmin
        self.attrs["kmax"] = kmax
        self.attrs["dk"] = dk
        return self.power

    def save(self, filename):
        import joblib

        save_dict = {
            "power": self.power,
            "attrs": self.attrs,
        }
        joblib.dump(save_dict, filename)

    @classmethod
    def load(cls, filename):
        import joblib

        load_dict = joblib.load(filename)
        self = FFTPowerCPP(
            load_dict["attrs"]["Nmesh"],
            load_dict["attrs"]["BoxSize"],
            load_dict["attrs"]["shotnoise"],
        )
        self.power = load_dict["power"]
        self.attrs = load_dict["attrs"]
        return self


class Mesh:
    def __init__(self, Nmesh, BoxSize):
        if isinstance(Nmesh, int) or isinstance(Nmesh, float):
            self.Nmesh = np.array([Nmesh, Nmesh, Nmesh], dtype=np.int32)
        else:
            self.Nmesh = Nmesh
        if isinstance(BoxSize, float):
            self.BoxSize = np.array([BoxSize, BoxSize, BoxSize])
        else:
            self.BoxSize = BoxSize

        self.attrs = {"Nmesh": self.Nmesh, "BoxSize": self.BoxSize}
        self.is_run = False
        self.real_field = None
        self.complex_field = None
        self.mesh = ToMesh(self.Nmesh, self.BoxSize)
        self.ndim = 3

    def is_structured_array(self, arr):
        """
        Test if the input array is a structured array
        by testing for `dtype.names`
        """
        if not isinstance(arr, np.ndarray) or not hasattr(arr, "dtype"):
            return False
        return arr.dtype.char == "V"

    def save(self, output_dir, mode="real"):
        if not self.is_run:
            raise ValueError("Mesh must run cic before saving")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if mode == "real":
            self.attrs["field_type"] = "real"
            np.save(output_dir + "/real_field.npy", self.real_field)
        elif mode == "complex":
            self.attrs["field_type"] = "complex"
            np.save(output_dir + "/complex_field.npy", self.complex_field)
        else:
            raise ValueError("mode must be real or complex")
        joblib.dump(self.attrs, output_dir + "/attrs_dict.pkl")

    @classmethod
    def load(cls, input_dir, mode="real"):
        attrs_dict = joblib.load(input_dir + "/attrs_dict.pkl")
        if mode == "real":
            real_field = np.load(input_dir + "/real_field.npy")
        elif mode == "complex":
            complex_field = np.load(input_dir + "/complex_field.npy")
        else:
            raise ValueError("mode must be real or complex")
        self = Mesh(attrs_dict["Nmesh"], attrs_dict["BoxSize"])
        self.attrs.update(attrs_dict)
        self.is_run = True
        if mode == "real":
            self.real_field = real_field
        elif mode == "complex":
            self.complex_field = complex_field
        else:
            pass
        return self

    def run_cic(
        self,
        data,
        position="Position",
        weight=None,
        field_dtype=np.float32,
        norm=False,
        nthreads=1,
    ):
        if not isinstance(data, dict):
            if not self.is_structured_array(data):
                raise ValueError(
                    (
                        "input data must have a "
                        "structured data type"
                    )
                )

        # compute the data type
        if hasattr(data, "dtype"):
            keys = sorted(data.dtype.names)
        else:
            keys = sorted(data.keys())

        if position not in keys:
            raise ValueError(
                (
                    "input data to ArrayCatalog must have a "
                    "structured data type with Position"
                )
            )
        else:
            position = data[position]
            N = data.shape[0]

        if weight is None or weight not in keys:
            weight = np.array([], dtype=position.dtype)
            W = N
            W2 = N
        else:
            weight = data[weight]
            W = np.sum(weight)
            W2 = np.sum(weight**2)
        self.attrs["N"] = N
        self.attrs["W"] = W
        self.attrs["W2"] = W2
        shotnoise = np.prod(self.BoxSize) * W2 / W**2
        self.attrs["shotnoise"] = shotnoise
        field = np.zeros(self.attrs["Nmesh"], dtype=field_dtype)

        self.mesh.RunCIC(position, weight, field, nthreads)
        self.is_run = True
        self.attrs["num_per_cell"] = W / np.prod(self.attrs["Nmesh"])

        if norm:
            self.real_field = field / self.attrs["num_per_cell"]
            return self.real_field
        else:
            self.real_field = field
            return self.real_field

    def r2c(self, field, compensated=False, nthreads=1):
        complex_filed = rfftn(field) / self.attrs["Nmesh"].astype(float).prod()
        self.attrs["compensated"] = compensated
        if compensated:
            freq_list = [
                np.fft.fftfreq(n=self.Nmesh[0], d=1.0).astype(np.complex64)
                * 2.0
                * np.pi,
                np.fft.fftfreq(n=self.Nmesh[1], d=1.0).astype(np.complex64)
                * 2.0
                * np.pi,
                np.fft.fftfreq(n=self.Nmesh[2], d=1.0).astype(np.complex64)[
                    : complex_filed.shape[2]
                ]
                * 2.0
                * np.pi,
            ]
            for i in range(self.ndim):
                if freq_list[i].shape[0] != complex_filed.shape[i]:
                    raise ValueError(
                        "The shape of the field and the frequency array are not consistent"
                    )
            self.mesh.DoCompensated(
                complex_filed,
                freq_x=freq_list[0],
                freq_y=freq_list[1],
                freq_z=freq_list[2],
                processors=nthreads,
            )
            self.complex_field = complex_filed
            return self.complex_field
        else:
            self.complex_field = complex_filed
            return self.complex_filed
