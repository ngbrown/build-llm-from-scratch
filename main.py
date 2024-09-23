def check_cuda():
    import torch
    print(f'Cuda is available: {torch.cuda.is_available()}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    check_cuda()
