numeros = [2,2,3,4,4,1,3,2]
tres = [5,6,7]

print(reversed(numeros))

#numeros.reverse()

for i in range(len(numeros)):

    if i < len(numeros) - 1 and numeros[i] == numeros[i + 1]: #and numeros[i] <:
        numeros[i+1] = numeros[i+1] + 1#numeros[i+1]

    print(numeros[i])


print(numeros)