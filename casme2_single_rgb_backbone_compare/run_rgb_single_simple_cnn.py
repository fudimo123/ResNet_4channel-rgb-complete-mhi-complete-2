import single_rgb_train as st

st.MODEL_NAME = 'simple_cnn'
st.RESULT_DIR = 'fusion_result_casme2_rgb_single_simple_cnn'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting CASME2 RGB single-channel experiment: simple_cnn")
    print("Results will be saved to: fusion_result_casme2_rgb_single_simple_cnn")
    st.main()


if __name__ == '__main__':
    main()
