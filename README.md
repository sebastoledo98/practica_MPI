# Practica #6: Comunicacion mediante MPI
Practica de Computacion Paralela sobre la implementación de aplicaciones de cómputo paralelo utilizando MPI

Python v3.12
OpenMPI v4.1.6

Se utilizo la herramienta [uv](https://github.com/astral-sh/uv) para manejar las diferentes versiones de python

El comando para ejecutar el proyecto es

```{bash}
mpiexec --mca btl_tcp_if_include 192.168.1.0/24 -hostfile hosts -n 3 ~/mpi_venv/bin/python ~/cluster_nlp/main_gui.py
```
