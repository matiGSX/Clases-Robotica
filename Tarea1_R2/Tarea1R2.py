import cv2
import numpy as np
import math

image=cv2.imread("Right_Arrow.png")

imageHSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

minH=np.array([0,0,0])
maxH=np.array([5,5,5])#el negro en HSV es "0 0 0"
binaryImage=cv2.inRange(imageHSV,minH,maxH)

cv2.imwrite("Right_Arrow_binary.png",binaryImage)

# cv2.imshow("original",cv2.resize(image,(800,600)))
#cv2.imshow("binary",cv2.resize(binaryImage,(800,600)))

#Calculo de los momentos
moments=cv2.moments(binaryImage,True);

#El area de la imagen
areaObject=moments['m00']
print("El area del objeto es:",areaObject)

#area del frame
areaImage=binaryImage.shape[0]*binaryImage.shape[1]

print("El area del frame es:", areaImage)

#posicion centro de masa
xcenter=moments['m10']/moments['m00']
ycenter=moments['m01']/moments['m00']

print("Area normalizada:",areaObject/areaImage)

#Posiciones normalizadas:
xnorm = xcenter/binaryImage.shape[0]
ynorm = ycenter/binaryImage.shape[1]
print("POSICION DEL CENTRO DE MASA (x,y):", xcenter/binaryImage.shape[0],ycenter/binaryImage.shape[1])


image_centro = cv2.circle(image,(int(xcenter), int(ycenter)), 5, (0,0,255), 2)

cv2.putText(image_centro, f"(x:{xnorm}, y:{ynorm})" ,(int(xcenter),int(ycenter+20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
#cv2.imshow("imagen",cv2.resize(image_centro,(800,600)))

#Para calcular la matriz de covarianza para PCA:
data_points = []
threshold = 0
H = binaryImage.shape[0]
W = binaryImage.shape[1]

for i in range(H):
    for j in range(W):
        if binaryImage[i, j] > threshold:
            data_points.append([i, j])
data_points = np.asarray(data_points)
cov_mat = np.cov(data_points.T)

#Eigenvectors y eigenvalues de la matriz de covarianza

eig_val, eig_vec = np.linalg.eig(cov_mat)
#eig_val es un vector de los eigenvalues
#eig_vect es una matriz cuyas columnas son los eigenvectores

#print ("eigenvector", eig_vec)
#print (eig_val)
#print(eig_val.max())

#Posicion del mayor eigenvalue
pos= np.where(eig_val==max(eig_val))[0]
print(pos)

#El componente principal será el eigenvector que corresponde al mayor eigenvalue,
PC1=eig_vec[:,pos] #Componente principal, vector normalizado
print("Componente Principal:", PC1)

#Para hallar al angulo que forma con el eje x:
angulo = math.atan((PC1[0]/PC1[1]))
angulo = angulo*180/np.pi

print("EL ANGULO ES:", angulo )

#Para la recta en la direccion de la componente principal:
m= math.tan(angulo*np.pi/180) #pendiente
y1=m*(0-xcenter)+ycenter #para x=0 P1(0,y1)
x2= (0-ycenter)/m+ xcenter #para y=0 P2(x2,0)

pt1= (0,int(y1))
pt2=(int(x2),0)

result=cv2.line(image_centro,pt1,pt2,(255,100,100),3)
cv2.imshow("Resultado",cv2.resize(result,(800,600)))
cv2.imwrite("Flechita_con_CM_y_Direccion.png",cv2.resize(result,(800,600)))

#Sistema de referencia:
# 0---------------> X
# |
# |
# |
# |
# |
# |
# v Y

key=cv2.waitKey(0)
cv2.destroyAllWindows()