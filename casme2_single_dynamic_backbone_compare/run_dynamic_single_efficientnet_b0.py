import single_dynamic_train as st

st.MODEL_NAME = 'efficientnet_b0'
st.RESULT_DIR = 'fusion_result_casme2_dynamic_single_efficientnet_b0'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting CASME2 dynamic single-channel experiment: efficientnet_b0")
    print("Results will be saved to: fusion_result_casme2_dynamic_single_efficientnet_b0")
    st.main()


if __name__ == '__main__':
    main()
