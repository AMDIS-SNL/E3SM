/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_CONTEXT_HPP
#define HOMMEXX_CONTEXT_HPP

#include <any>
#include <string>
#include <map>
#include <memory>
#include <functional>

#include <ekat_assert.hpp>

namespace Homme {

/* A Context manages resources that are morally singletons. Context is
 * meant to have two roles. First, a Context singleton is the only singleton in
 * the program. Second, a context need not be a singleton, and each Context
 * object can have different Elements, ReferenceElement, etc., objects. (That
 * probably isn't needed, but Context immediately supports it.)
 *
 * Finally, Context has two singleton functions: singleton(), which returns
 * Context&, and finalize_singleton(). The second is called in a unit test exe
 * main before Kokkos::finalize().
 */
class Context {
public:

  // Getters for a managed object.
  template<typename ConcreteType>
  bool has () const;

  // Setters for a managed object.
  template<typename ConcreteType, typename... Args>
  ConcreteType& create (Args&&... args);

  template<typename ConcreteType>
  void create_ref (ConcreteType& src);

  // More relaxed than create, it won't throw if the
  // object already exists, and simply return it.
  // NOTE: this is allows more flexibility in the cxx-f90
  //       interfaces, in that you have more freedom in
  //       the order of certain function calls.
  //       With 'create', you have to either check if an
  //       object was already created (and create it if it wasn't),
  //       or make sure that the function that calls 'create<T>'
  //       is *always* invoked *before* those that call 'get<T>'.
  template<typename ConcreteType, typename... Args>
  ConcreteType& create_if_not_there (Args&&... args);

  // Getters for a managed object.
  template<typename ConcreteType>
  ConcreteType& get () const;

  template<typename ConcreteType>
  std::shared_ptr<ConcreteType> get_ptr () const;

  // Exactly one singleton.
  static Context& singleton() {
    static Context c;
    return c;
  }

  static void finalize_singleton() {
    singleton().clear();
  }

  const std::map<std::string,std::any>& any_map () const {
    return m_any_map;
  }
  std::map<std::string,std::any>& any_map () {
    return m_any_map;
  }
private:

  std::map<std::string,std::any> m_any_map;
  std::map<std::string,std::any> m_members;
  std::map<std::string, bool>    m_is_ref_wrapper;

  // Clear the objects Context manages.
  void clear() {
    m_members.clear();
    m_any_map.clear();
    m_is_ref_wrapper.clear();
  }
};

// ==================== IMPLEMENTATION =================== //

template<typename ConcreteType>
bool Context::has () const {
  const std::string& name = typeid(ConcreteType).name();
  auto it = m_members.find(name);
  return it!=m_members.end();
}

template<typename ConcreteType, typename... Args>
ConcreteType& Context::create_if_not_there (Args&&... args)
{
  const std::string name = typeid(ConcreteType).name();
  auto it = m_members.find(name);
  if (it==m_members.end()) {
    auto ptr = std::make_shared<ConcreteType>(std::forward<Args>(args)...);
    m_members[name] = ptr;
    m_is_ref_wrapper[name] = false;
    return *ptr;
  }
  auto ptr = *std::any_cast<std::shared_ptr<ConcreteType>>(&it->second);
  return *ptr;
}

template<typename ConcreteType, typename... Args>
ConcreteType& Context::create (Args&&... args)
{
  const std::string name = typeid(ConcreteType).name();
  EKAT_REQUIRE_MSG(!has<ConcreteType>(),
      "Error! An object for the concrete type " + name +
      " is already stored. The 'Context' class does not allow overwriting or duplicates.\n");

  auto ptr = std::make_shared<ConcreteType>(std::forward<Args>(args)...);
  m_members[name] = ptr;
  m_is_ref_wrapper[typeid(ConcreteType).name()] = false;

  return *ptr;
}

template<typename ConcreteType>
void Context::create_ref (ConcreteType& src)
{
  const std::string name = typeid(ConcreteType).name();
  EKAT_REQUIRE_MSG(!has<ConcreteType>(),
      "Error! An object for the concrete type " + name +
      " is already stored. The 'Context' class does not allow overwriting or duplicates.\n");

  m_members[name] = std::reference_wrapper<ConcreteType>(src);
  m_is_ref_wrapper[name] = true;
}

template<typename ConcreteType>
ConcreteType& Context::get () const
{
  const std::string name = typeid(ConcreteType).name();
  auto it = m_members.find(name);
  EKAT_REQUIRE_MSG(it!=m_members.end(), "Error! Context member '" + name + "' not found.\n");
  if (m_is_ref_wrapper.at(name)) {
    return std::any_cast<std::reference_wrapper<ConcreteType>>(it->second).get();
  } else {
    return *std::any_cast<std::shared_ptr<ConcreteType>>(it->second);
  }
}

template<typename ConcreteType>
std::shared_ptr<ConcreteType> Context::get_ptr() const
{
  const std::string& name = typeid(ConcreteType).name();
  auto it = m_members.find(name);
  EKAT_REQUIRE_MSG(it!=m_members.end(), "Error! Context member '" + name + "' not found.\n");
  EKAT_REQUIRE_MSG(!m_is_ref_wrapper.at(name),
      "Error! Context member '" + name + "' is only available as a reference.\n");

  return std::any_cast<std::shared_ptr<ConcreteType>>(it->second);
}

} // namespace Homme

#endif // HOMMEXX_CONTEXT_HPP
