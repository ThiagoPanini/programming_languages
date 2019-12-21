import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy

IM = img.imread("imagem.tiff")
IM = numpy.matrix(IM)/255
plt.imshow(IM, cmap=plt.cm.gray, interpolation='none')
plt.show()

# decomposicao SVD
(W,S,C)= numpy.linalg.svd(IM)

IM2 =  W*numpy.diag(S)*C
plt.imshow(IM2, cmap=plt.cm.gray, interpolation='none')
plt.show()

print (S)
S[4:] = 0
print (S)

IM2 =  W*numpy.diag(S)*C
plt.imshow(IM2, cmap=plt.cm.gray, interpolation='none')
plt.show()


#plt.imshow(W, cmap=plt.cm.gray, interpolation='none')
#plt.show() 
#plt.imshow(numpy.diag(S), cmap=plt.cm.gray, interpolation='none')
#plt.show() 
#plt.imshow(C, cmap=plt.cm.gray, interpolation='none')
#plt.show() 



