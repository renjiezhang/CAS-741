# 3D Box class
# Author: Mustafa Haddara

# This class represents a rectangle as (x,y,z) coordinate representing the 
# front-facing top-left corner, and a (w,h,d) tuple representing the width,
# height, depth
class Box3D(object):

    # Faces enums
    FRONT = 0
    TOP = 1
    SIDE = 2

    # Constructor accepts 2 parameters
    # corner: (x,y,z) tuple for the top left corner
    # size: (w,h,d) tuple containing the width and height
    def __init__(self, corner, size):
        if len(corner) != 3 or len(size) != 3:
            raise ValueError("Invalid argument size")
        # double underscore (__) before the identifier is python convention
        # for private variables
        self.__frontTopLeftCorner = corner
        self.__dimensions = size

    def getFrontTopLeftCorner(self):
        return self.__frontTopLeftCorner

    def setFrontTopLeftCorner(self, newCorner):
        self.__frontTopLeftCorner = newCorner

    def getDimensions(self):
        return self.__dimensions

    def setDimensions(self, newSize):
        self.__dimensions = newSize

    def getDimensionsOfFace(self, face):
        if face == Box3D.FRONT:
            width,height = self.__dimensions[0],self.__dimensions[1] # w,h
        elif face == Box3D.TOP:
            width,height = self.__dimensions[0],self.__dimensions[2] # w,d
        elif face == Box3D.SIDE:
            width,height = self.__dimensions[1],self.__dimensions[2] # h,d
        else:
            raise ValueError("Invalid `face` argument")
        return width,height

    def getAreaOfFace(self, face):
        width,height = self.getDimensionsOfFace(face)
        return width * height

    def getPerimeterOfFace(self, face):
        width,height = self.getDimensionsOfFace(face)
        return 2 * (width + height)

    def getVolume(self):
        width,height,depth = self.__dimensions
        return width * height * depth

    def getSurfaceArea(self):
        surfaceArea = 0
        for face in (Box3D.FRONT, Box3D.TOP, Box3D.SIDE):
            surfaceArea += (2 * self.getAreaOfFace(face))
        return surfaceArea

# Example usage
if __name__ == '__main__':
    b = Box3D( (1,1,1), (2,2,2) )
    print b.getSurfaceArea()