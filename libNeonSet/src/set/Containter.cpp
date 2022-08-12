#include "Neon/set/Containter.h"
#include "Neon/set/ContainerTools/AnchorContainer.h"

namespace Neon::set {

auto Container::run(int            streamIdx,
                    Neon::DataView dataView)
    -> void
{
    mContainer->run(streamIdx, dataView);
}


auto Container::run(Neon::SetIdx   setIdx,
                    int            streamIdx,
                    Neon::DataView dataView)
    -> void
{
    mContainer->run(setIdx, streamIdx, dataView);
}

auto Container::getContainerInterface()
    -> Neon::set::internal::ContainerAPI&
{
    return mContainer.operator*();
}

auto Container::getContainerInterface() const
    -> const Neon::set::internal::ContainerAPI&
{
    return mContainer.operator*();
}


auto Container::getContainerInterfaceShrPtr()
    -> std::shared_ptr<Neon::set::internal::ContainerAPI>
{
    return mContainer;
}

auto Container::factoryDeviceThenHostManaged(const std::string& name,
                                             Container&         device,
                                             Container&         host) -> Container
{
    auto k = new Neon::set::internal::DeviceThenHostManagedContainer(name,
                                                                     device.getContainerInterfaceShrPtr(),
                                                                     host.getContainerInterfaceShrPtr());

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container(tmp);
}

auto Container::factoryAnchor(const std::string& name) -> Container
{
    auto                                               k = new Neon::set::internal::AnchorContainer(name);
    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container();
}

auto Container::getName() const
    -> const std::string&
{
    return mContainer->getName();
}

auto Container::getUid() const
    -> uint64_t
{
    const auto uid = (uint64_t)mContainer.get();
    return uid;
}

auto Container::logTokens()
    -> void
{
    return mContainer->toLog(getUid());
}

auto Container::getHostContainer() const
    -> Container
{
    std::shared_ptr<Neon::set::internal::ContainerAPI> hostAPI =
        mContainer->getHostContainer();
    return Container(hostAPI);
}

auto Container::getDeviceContainer() const -> Container
{
    std::shared_ptr<Neon::set::internal::ContainerAPI> deviceAPI =
        mContainer->getDeviceContainer();
    return Container(deviceAPI);
}

auto Container::getDataViewSupport() const
    -> Neon::set::internal::ContainerAPI::DataViewSupport
{
    auto&      api = this->getContainerInterface();
    auto const dwSupport = api.getDataViewSupport();
    return dwSupport;
}

auto Container::getContainerType() const
    -> Neon::set::internal::ContainerType
{
    auto&      api = this->getContainerInterface();
    auto const type = api.getContainerType();
    return type;
}

Container::Container(std::shared_ptr<Neon::set::internal::ContainerAPI>& container)
    : mContainer(container)
{
    // Empty
}

Container::Container(std::shared_ptr<Neon::set::internal::ContainerAPI>&& container)
    : mContainer(container)
{
    // Empty
}


}  // namespace Neon::set