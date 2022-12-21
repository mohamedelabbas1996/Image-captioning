import tqdm


def train(dataloader, optim, model, criterion, n_epochs, device, loss=0):
    model.train()
    for epoch in range(n_epochs):
        with tqdm.tqdm(dataloader, unit="batch") as tepoch:
            losses = loss
            for idx, (image, caption, target) in enumerate(tepoch):
                image, caption, target = image.to(device), caption.to(device), target.to(device)
                target = target.view(-1)
                # print ("target dim", target.shape)
                model = model.train()
                tepoch.set_description(f"Epoch {epoch}")
                optim.zero_grad()
                output = model(image, caption)
                # print ("caption shape", caption.shape)
                # print ("output shape", output.shape)
                # print(output.shape, output_token.shape )
                # print (output.dtype , target_token.dtype , output.shape, target_token.shape)
                loss = criterion(output, target)
                losses += loss.item()
                tepoch.set_postfix(loss=f"{losses / (idx + 1):0.4f}")
                loss.backward()
                optim.step()
