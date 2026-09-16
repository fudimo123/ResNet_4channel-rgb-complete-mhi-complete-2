import single_dynamic_train as st

st.MODEL_NAME = 'plain_resnet18'
st.RESULT_DIR = 'fusion_result_casme2_dynamic_single_plain_resnet18'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting CASME2 dynamic single-channel experiment: plain_resnet18")
    print("Results will be saved to: fusion_result_casme2_dynamic_single_plain_resnet18")
    st.main()


if __name__ == '__main__':
    main()
