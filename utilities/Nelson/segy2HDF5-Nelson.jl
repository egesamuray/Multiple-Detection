using SeisIO
using PyPlot
using HDF5


# traceNum = 2*184

searchdir(path,key) = filter(x->contains(x,key), readdir(path))
files = searchdir("/data/NelsonData/raw-data",".segy");

for str in files
	block = segy_read(str);
	d = convert(Array{Float32,2}, block.data)
	h5write("/data/NelsonData/hdf5/NelsonShots.hdf5", str, transpose(d))
end


# imshow(d[:,1:2*traceNum], cmap="Greys", vmin=-300, vmax=300)
