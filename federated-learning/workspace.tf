# Configure the AzureRM Provider
provider "azurerm" {
  features {
  }
}

# Dependent resources for Azure Machine Learning
resource "azurerm_application_insights" "default" {
  count               = 3
  name                = "medical-imaging-rafl-appi${count.index}"
  location            = var.location
  resource_group_name = var.resource_group_name
  application_type    = "web"
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "azurerm_key_vault" "default" {
  count                    = 3
  name                     = "${var.prefix}${var.environment}${random_integer.suffix.result}kv${count.index}"
  location                 = var.location
  resource_group_name      = var.resource_group_name
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "premium"
  purge_protection_enabled = false
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "azurerm_storage_account" "default" {
  count                           = 3
  name                            = "${var.prefix}${var.environment}${random_integer.suffix.result}st${count.index}"
  location                        = var.location
  resource_group_name             = var.resource_group_name
  account_tier                    = "Standard"
  account_replication_type        = "GRS"
  allow_nested_items_to_be_public = false
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

resource "azurerm_container_registry" "default" {
  count               = 3
  name                = "${var.prefix}${var.environment}${random_integer.suffix.result}cr${count.index}"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "Premium"
  admin_enabled       = true
  tags = {
    contact  = var.contact
    customer = var.customer
  }
}

# Machine Learning workspace
resource "azurerm_machine_learning_workspace" "default" {
  count                         = 3
  name                          = "medical-imaging-rafl-mlw${count.index}"
  location                      = var.location
  resource_group_name           = var.resource_group_name
  application_insights_id       = azurerm_application_insights.default[count.index].id
  key_vault_id                  = azurerm_key_vault.default[count.index].id
  storage_account_id            = azurerm_storage_account.default[count.index].id
  container_registry_id         = azurerm_container_registry.default[count.index].id
  public_network_access_enabled = true
  tags = {
    contact  = var.contact
    customer = var.customer
  }

  identity {
    type = "SystemAssigned"
  }
}

# compute instances
resource "azurerm_machine_learning_compute_instance" "example" {
  count                         = 3
  name                          = "rafl-ci${count.index}"
  location                      = var.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.default[count.index].id
  virtual_machine_size          = "STANDARD_DS2_V2"
  authorization_type            = "personal"
  ssh {
    public_key = ""
  }
  subnet_resource_id = azurerm_subnet.internal.id
  description        = "foo"
  tags = {
    foo = "bar"
  }
}
