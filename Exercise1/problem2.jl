using Images  # Basic image processing functions
using PyPlot  # Plotting and image loading
using JLD     # Functions for loading and storing data in the ".jld" format

# Load the Bayer image from the provided .jld file
function loadbayer()

  data=load("../data-julia/bayerdata.jld","bayerimg")
  data=data./255
  return data::Array{Float64,2}
end

# Seperate the Bayer image into three images (one for each color channel), filling up all
# unknown values with 0
function separatebayer(data::Array{Float64,2})
  r= zeros(data)
  g= zeros(data)
  b= zeros(data)
 r[1:2:end,2:2:end] = data[1:2:end,2:2:end]
 g[1:2:end,1:2:end] = data[1:2:end,1:2:end]
 g[2:2:end,2:2:end] = data[2:2:end,2:2:end]
 b[2:2:end,1:2:end] = data[2:2:end,1:2:end]
  return r::Array{Float64,2}, g::Array{Float64,2}, b::Array{Float64,2}
end

# Combine three colorchannels into a single image
function makeimage(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
  image=cat(3,r,g,b)
  return image::Array{Float64,3}
end

# Interpolate missing color values using bilinear interpolation
function debayer(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
  f1 = [0.25 0.5 0.25; 0.5 1 0.5; 0.25 0.5 0.25]
  f2 = [0 0.25 0; 0.25 1 0.25; 0 0.25 0]
   r = imfilter(r, f1, "symmetric")
   g = imfilter(g, f2, "symmetric")
   b = imfilter(b, f1, "symmetric")
   image=cat(3,r,g,b)
  return image::Array{Float64,3}
end

# display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})
  figure()
  subplot(1,2,1)
  imshow(img1)
  axis("off")
  subplot(1,2,2)
  imshow(img2)
  axis("off")
  return nothing
end

#= Problem 1
Warm-Up / Bayer Interpolation =#

function problem1()
  # load imgage
  data = loadbayer()

  # seperate date
  r,g,b = separatebayer(data)

  # merge raw bayer
  img1 = makeimage(r,g,b)

  # interpolate bayer
  img2 = debayer(r,g,b)

  # display images
  displayimages(img1, img2)
  return
end
problem1()
