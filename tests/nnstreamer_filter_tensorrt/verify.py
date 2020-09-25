import struct

tfile = "./sj.out.log"

def readbyte (fname):
	f = open(fname, 'rb')
	rbyte = f.read()
	f.close()
	return rbyte

rbyte = readbyte (tfile)
print(len(rbyte))
softmax = []
for i in range(10):
	byte = b''
	byte += struct.pack("<B", rbyte[i * 4])
	byte += struct.pack("<B", rbyte[i * 4 + 1])
	byte += struct.pack("<B", rbyte[i * 4 + 2])
	byte += struct.pack("<B", rbyte[i * 4 + 3])
	softmax.append(struct.unpack('f', byte))

print(softmax)
print(max(softmax))


