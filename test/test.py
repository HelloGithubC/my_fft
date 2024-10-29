import numpy as np 
from my_fft import FFTPower, Mesh 

data = np.load("test_data.pkl.npy")
print(data.shape)
print(data.dtype)

mesh = Mesh(Nmesh=100, BoxSize=1000.0)
cic_mesh = mesh.run_cic(data, weight="Weight", norm=False, nthreads=3) # For PS of overdensity, you can set norm=True
mesh_complex = mesh.r2c(cic_mesh, compensated=True, nthreads=2)

print(mesh.attrs)

fftpower = FFTPower(Nmesh=100, BoxSize=1000.0, shotnoise=mesh.attrs["shotnoise"]) # Nmesh and BoxSize you need keep the same as Mesh you used
power = fftpower.run(mesh_complex, kmin=0.1, kmax=0.5, dk=0.02, mode="1d", nthreads=2, Nmu=5, force_conj=False) # mode can be 1d and 2d. force_conj is False usually. See Readme.md for more details.
fftpower.save("test_power_1D.pkl")

power = fftpower.run(mesh_complex, kmin=0.1, kmax=0.5, dk=0.02, mode="2d", nthreads=2, Nmu=5, force_conj=False) # mode can be 1d and 2d. force_conj is False usually. See Readme.md for more details.
fftpower.save("test_power_2D.pkl")