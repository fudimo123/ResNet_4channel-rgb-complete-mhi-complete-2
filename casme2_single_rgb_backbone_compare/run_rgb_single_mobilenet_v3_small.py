import single_rgb_train as st

st.MODEL_NAME = 'mobilenet_v3_small'
st.RESULT_DIR = 'fusion_result_casme2_rgb_single_mobilenet_v3_small'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting CASME2 RGB single-channel experiment: mobilenet_v3_small")
    print("Results will be saved to: fusion_result_casme2_rgb_single_mobilenet_v3_small")
    st.main()


if __name__ == '__main__':
    main()
