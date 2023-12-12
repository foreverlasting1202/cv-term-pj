for i in {1..50}
do
   python train.py
done

MESA_GL_VERSION_OVERRIDE=4.1 python run.py
