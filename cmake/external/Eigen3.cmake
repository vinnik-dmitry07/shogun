set(EIGEN_INCLUDE_DIR ${THIRD_PARTY_INCLUDE_DIR}/eigen)
set(EIGEN_VERSION 3.3.4)
include(ExternalProject)
ExternalProject_Add(
	Eigen3
	PREFIX ${CMAKE_BINARY_DIR}/Eigen3
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/Eigen3
	URL https://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.bz2
	URL_MD5 a7aab9f758249b86c93221ad417fbe18
	CMAKE_ARGS -DINCLUDE_INSTALL_DIR:STRING=${EIGEN_INCLUDE_DIR}
		-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}${CMAKE_DEFINITIONS}
		-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}${CMAKE_DEFINITIONS}
		-DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
		-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
		-DCMAKE_BUILD_TYPE:STRING=Release
		-DEIGEN_BUILD_PKGCONFIG=0
	)

ADD_SHOGUN_DEPENDENCY(Eigen3)

UNSET(C_COMPILER)
UNSET(CXX_COMPILER)
